"""命令行入口：多轮对话与流式输出，复用 FAQ 与 RAG 双通道。"""

import warnings

warnings.filterwarnings("ignore")
from mysql_qa import MySQLClient, RedisClient, BM25Search
from rag_qa import VectorStore, RAGSystem

from base.config import config
from base.logger import logger

from openai import OpenAI
import time
import pymysql
import uuid


class IntegratedQASystem:
    """
    集成问答系统(V2).
    流程: 取历史 -> BM25 查 MySQL -> 命中就直出 -> 没命中走 RAG(带历史) -> 存历史
    """

    def __init__(self):
        self.logger = logger
        self.config = config

        # 把已创建的客户端注入 BM25，避免重复建立 MySQL/Redis 连接。
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

        try:
            self.client = OpenAI(
                api_key=self.config.DASHSCOPE_API_KEY,
                base_url=self.config.DASHSCOPE_BASE_URL,
            )
        except Exception as e:
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            raise

        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)
        self.init_conversation_table()

    def init_conversation_table(self):
        """
        在 MySQL 里建 conversations 表, 用来存多轮对话.
        表结构: 每条记录 = 一次问答(会话ID + 问题 + 答案 + 时间).
        """
        # 没有可用连接就直接放弃多轮功能, 而不是崩溃.
        # 场景: MySQL 没启动, 你的 MySQLClient 会降级去读本地 CSV, 此时 cursor 是 None.
        if self.mysql_client.cursor is None:
            self.logger.warning("MySQL 不可用(已降级为本地CSV), 多轮对话历史功能已关闭")
            self.history_enabled = False
            return

        try:
            self.mysql_client.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id         INT AUTO_INCREMENT PRIMARY KEY,   -- 主键
                    session_id VARCHAR(36) NOT NULL,             -- 会话ID, 一次对话一个
                    question   TEXT NOT NULL,                    -- 用户问题
                    answer     TEXT NOT NULL,                    -- 系统答案
                    timestamp  DATETIME NOT NULL,                -- 时间
                    INDEX idx_session_time (session_id, timestamp, id) -- 按会话和时间查历史
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            self.mysql_client.conn.commit()
            self.history_enabled = True
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"初始化对话历史表失败: {e}")
            # 表建不出来不该让整个系统起不来, 降级即可
            self.history_enabled = False

    def call_dashscope(self, prompt):
        """
        RAG 内部会拿这个函数去调模型(比如 HyDE 生成假设答案、子问题拆分).
        这些中间步骤必须拿到"完整字符串", 所以这里不能开 stream.
        :return: 模型返回的完整文本
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个靠谱的助手, 根据给的信息好好回答问题。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
            )
            return (
                completion.choices[0].message.content
                if completion.choices
                else "错误：无效的 LLM 响应"
            )
        except Exception as e:
            self.logger.error(f"LLM 调用失败: {e}")
            return f"错误：LLM 调用失败 - {e}"

    def call_dashscope_stream(self, prompt):
        """
        流式调用: 一小段一小段地返回内容(yield).
        注意: 这是个"生成器函数", 调用它不会立刻执行, 必须 for 循环才真正发请求.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"流式调用失败: {e}")
            yield f"错误：LLM 调用失败 - {e}"

    def _fetch_recent_history(self, session_id):
        """
        从 conversations 表取该会话最近的 5 条问答, 按时间正序返回.
        :return: [{'question': ..., 'answer': ...}, ...]
        """
        if not getattr(self, "history_enabled", False) or not session_id:
            return []
        try:
            self.mysql_client.cursor.execute(
                """
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                -- 【优化】id 作为并列时间戳时的稳定排序键，保证历史顺序可预测。
                ORDER BY timestamp DESC, id DESC
                LIMIT %s
                """,
                (session_id, 5),
            )
            rows = self.mysql_client.cursor.fetchall()
            history = [{"question": r[0], "answer": r[1]} for r in rows]
            # 上面是按时间倒序取的(取最近的), 返回前要翻回正序, 否则模型读到的是倒着的历史
            return history[::-1]
        except pymysql.MySQLError as e:
            self.logger.error(f"获取对话历史失败: {e}")
            return []

    def get_session_history(self, session_id):
        return self._fetch_recent_history(session_id)

    def update_session_history(self, session_id: str, question: str, answer: str):
        if not getattr(self, "history_enabled", False) or not session_id:
            return []
        try:
            self.mysql_client.cursor.execute(
                """
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
                """,
                (session_id, question, answer),
            )

            self.mysql_client.cursor.execute(
                """
                DELETE FROM conversations
                WHERE session_id = %s
                  AND id NOT IN (
                      SELECT id FROM (
                          SELECT id FROM conversations
                          WHERE session_id = %s
                           -- 【优化】清理历史时使用与查询一致的稳定排序。
                           ORDER BY timestamp DESC, id DESC
                          LIMIT %s
                      ) AS sub
                  )
                """,
                (session_id, session_id, 5),
            )
            self.mysql_client.conn.commit()
            self.logger.info(f"会话 {session_id} 历史已更新")
            return self._fetch_recent_history(session_id)
        except pymysql.MySQLError as e:
            self.logger.error(f"更新会话历史失败: {e}")
            self.mysql_client.conn.rollback()
            return []

    def clear_session_history(self, session_id: str) -> bool:
        if not getattr(self, "history_enabled", False) or not session_id:
            return False
        try:
            self.mysql_client.cursor.execute(
                "DELETE FROM conversations WHERE session_id = %s", (session_id,)
            )
            self.mysql_client.conn.commit()
            self.logger.info(f"会话 {session_id} 历史已清空")
            return True
        except pymysql.MySQLError as e:
            self.logger.error(f"清空历史失败: {e}")
            self.mysql_client.conn.rollback()
            return False

    def query(self, query, source_filter=None, session_id=None):
        """
        为什么用 yield 而不是 return?
        因为要让调用方(命令行 / 前端)边收边显示. 每 yield 一次, 调用方就能立刻拿到一段.
        yield 的二元组含义: (要显示的内容, 本次是否为最后一包)
        """
        start_time = time.time()
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")
        history = self.get_session_history(session_id) if session_id else []

        # search() 只返回一个值, 用 None 判断是否要回退 RAG
        answer = self.bm25_search.search(query, threshold=0.85)

        if answer:
            self.logger.info(f"MySQL 命中答案: {answer}")
            if session_id:
                self.update_session_history(session_id, query, answer)
            self.logger.info(f"处理耗时 {time.time() - start_time:.2f} 秒")
            yield answer, True
            return

        self.logger.info("MySQL 未命中, 回退到 RAG")
        collected = ""  # 攒完整答案, 存历史要用完整版, 不能用碎片
        result = self.rag_system.generate_answer(
            query, source_filter=source_filter, history=history
        )

        if isinstance(result, str):
            # RAGSystem.generate_answer 可能返回字符串, 也可能返回生成器, 这里兼容两种:
            #   - 字符串 -> 一次性吐出, 标 is_complete=True
            #   - 生成器 -> 逐段吐出
            collected = result
            yield result, True
        else:
            for token in result:
                collected += token
                yield token, False
            yield "", True

        if session_id:
            self.update_session_history(session_id, query, collected)
        self.logger.info(f"处理耗时 {time.time() - start_time:.2f} 秒")


def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("\n欢迎使用集成问答系统 V2 (多轮对话 + 流式输出)")
    print(f"会话ID: {session_id}")
    print(f"支持的学科类别: {qa_system.config.VALID_SOURCES}")
    print("输入问题回车提问, 输入 exit 退出")

    try:
        while True:
            query = input("\n请输入问题: ").strip()
            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见!")
                break
            if not query:
                continue

            source_filter = input("学科类别(回车跳过): ").strip()
            if source_filter not in qa_system.config.VALID_SOURCES:
                source_filter = None

            print("\n答案: ", end="", flush=True)
            for token, is_complete in qa_system.query(
                query, source_filter=source_filter, session_id=session_id
            ):
                if token:
                    print(token, end="", flush=True)
                if is_complete:
                    print()
                    break

            history = qa_system.get_session_history(session_id)
            if history:
                print("\n--- 最近对话历史 ---")
                for i, item in enumerate(history, 1):
                    print(f"{i}. 问: {item['question']}")
                    print(f"   答: {item['answer'][:60]}...")
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        qa_system.mysql_client.close()

if __name__ == "__main__":
    main()
