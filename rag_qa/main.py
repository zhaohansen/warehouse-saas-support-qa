"""数据与查询入口：批量建索引，或在命令行问答。"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Windows 控制台默认 GBK, 遇到生僻字符会抛 UnicodeEncodeError, 这里改为用替换字符输出
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

from base.config import config as conf
from base.logger import logger
from rag_qa.core.document_processor import process_documents
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from openai import OpenAI


def main(query_mode=True, directory_path=None):
    """
    函数功能: 系统主函数, 控制两种运行模式.
    :param query_mode:
        数据处理模式: query_model=False  解析指定目录的文档, 分割为文档块并存入向量数据.
        交互式查询模式: query_model=True 接收用户问题, 调用RAG系统生成答案并展示.
    :param directory_path: 数据处理模式下的文档根目录, 默认取配置里的 DATA_DIR (rag_qa/data), 需包含各个学科的子目录, 例如: ai_data, java_data
    :return: 无
    """

    # 未指定目录时, 使用配置中的数据目录 -> 避免依赖启动时的工作目录.
    directory_path = directory_path or conf.DATA_DIR

    if query_mode and not conf.DASHSCOPE_API_KEY:
        print("错误：未配置 DASHSCOPE_API_KEY。请在 base/config.ini 的 [llm] 中填写 api_key，或设置同名环境变量。")
        return

    try:
        client = OpenAI(api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL)
    except Exception as e:
        logger.error(f"初始化 OpenAI 客户端失败 (请检查 API Key 和 Base URL): {e}")
        if query_mode:
             print("错误：无法初始化语言模型客户端，无法进入查询模式。")
             return
        client = None

    def call_dashscope(prompt):
        if not client:
            logger.error("LLM 客户端未初始化，无法调用 call_dashscope")
            return f"错误: LLM客户端不可用"
        try:
            completion = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手."},
                    {"role": "user", "content": prompt},
                ]
            )
            if completion.choices and completion.choices[0].message:
                 return completion.choices[0].message.content
            else:
                 logger.error("LLM API 调用返回无效响应或空消息")
                 return "错误: LLM返回无效响应"
        except Exception as e:
            logger.error(f"LLM API (call_dashscope) 调用失败: {e}")
            return f"错误: 调用LLM失败 - {e}"

    try:
        vector_store = VectorStore(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME,
        )
    except Exception as e:
        logger.error(f"初始化 VectorStore 失败: {e}")
        print("错误：向量库初始化失败，请检查 Milvus 服务(默认 localhost:19530)以及 rag_qa/models 下的模型文件是否就绪。")
        return

    if not query_mode:
        logger.info("进入数据处理模式...")
        total_chunks_added = 0
        for source_dir in conf.VALID_SOURCES:
            dir_path = os.path.join(directory_path, f"{source_dir}_data")
            if os.path.exists(dir_path):
                logger.info(f"开始处理目录: {dir_path}")
                try:
                    chunks = process_documents(
                        dir_path,
                        conf.PARENT_CHUNK_SIZE,
                        conf.CHILD_CHUNK_SIZE,
                        conf.CHUNK_OVERLAP,
                    )
                    if chunks:
                        vector_store.add_documents(chunks)
                        total_chunks_added += len(chunks)
                        logger.info(f"成功处理目录 {dir_path}，添加了 {len(chunks)} 个文档块")
                    else:
                        logger.info(f"目录 {dir_path} 未发现有效文档或处理结果为空")
                except Exception as e:
                    logger.error(f"处理目录 {dir_path} 时出错: {e}")
            else:
                logger.warning(f"目录 {dir_path} 不存在，跳过处理")
        logger.info(f"数据处理完成，共添加了 {total_chunks_added} 个文档块到向量存储")
    else:
        if not client:
            print("错误：查询模式需要语言模型客户端，但初始化失败。")
            return

        logger.info("进入交互式查询模式...")
        try:
            rag_system = RAGSystem(vector_store, call_dashscope)
        except Exception as e:
             logger.error(f"初始化 RAGSystem 失败: {e}")
             print("错误：无法初始化 RAG 系统，无法进入查询模式。")
             return

        valid_sources = conf.VALID_SOURCES
        print("\n欢迎使用 EduRAG 交互式查询系统！")
        print(f"支持的学科类别：{valid_sources}")
        print("输入您的问题，或输入 'exit' 退出。")

        while True:
            query = input("\n请输入您的问题：")
            if query.lower() == "exit":
                logger.info("用户退出查询模式")
                print("再见！")
                break

            source_filter_input = input(f"请输入学科类别 ({'/'.join(valid_sources)}) (直接回车默认不过滤)：").strip()
            source_filter = None
            if source_filter_input:
                if source_filter_input in valid_sources:
                    source_filter = source_filter_input
                    logger.info(f"用户选择了学科过滤: {source_filter}")
                else:
                    logger.warning(
                        f"无效的学科类别 '{source_filter_input}'，将不过滤"
                    )
                    print(f"提示：输入的学科 '{source_filter_input}' 无效，将不过滤。")

            try:
                print("正在生成答案，请稍候...")
                answer = rag_system.generate_answer(query, source_filter=source_filter)
                print("-" * 30)
                print(f"问题: {query}")
                print(f"回答: {answer}")
                print("-" * 30)
            except Exception as e:
                logger.error(f"处理查询 '{query}' 时失败: {str(e)}")
                print(f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员。\n")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="EduRAG System Main Entry Point")
    parser.add_argument('--data-processing', action='store_true', help='Run in data processing mode instead of query mode.')
    parser.add_argument('--data-dir', type=str, default=conf.DATA_DIR, help='Path to the data directory.')
    args = parser.parse_args()
    main(query_mode=(not args.data_processing), directory_path=args.data_dir)
