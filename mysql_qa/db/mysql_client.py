"""串行访问 MySQL 常见问答；启动或运行时失败可回退到 CSV。"""
import csv
from pathlib import Path
import threading
import pymysql
from base import Config
from base.logger import logger


def _in_scope(records):
    """只保留配置中的学科，避免旧题库里的其它学科 FAQ 混入问答。"""
    allowed = {str(s).strip().casefold() for s in Config.VALID_SOURCES}
    return [r for r in records if str(r.get('source', '')).strip().casefold() in allowed]


class MySQLClient:
    def __init__(self):
        self.lock = threading.RLock()
        self.conn = self.cursor = None
        self.faq_source = None
        self.csv_path = Path(__file__).with_name('faq_knowledge_base.csv')
        try:
            self.conn = pymysql.connect(host=Config.mysql_host, user=Config.mysql_user,
                password=Config.mysql_password, db=Config.mysql_db, charset='utf8mb4',
                connect_timeout=3, read_timeout=10, write_timeout=10, autocommit=True)
            self.cursor = self.conn.cursor()
        except Exception:
            logger.warning('MySQL 不可用，FAQ 改读本地 CSV', exc_info=True)
            self._shutdown_connection()

    def _shutdown_connection(self):
        """逐个释放连接资源，确保一个 close 失败不影响另一个。"""
        cursor, conn = self.cursor, self.conn
        self.cursor = self.conn = None
        for resource in (cursor, conn):
            close = getattr(resource, 'close', None)
            if close:
                try: close()
                except Exception: logger.debug('关闭 MySQL 资源失败', exc_info=True)

    @staticmethod
    def _subject(value):
        return str(value).replace('学科', '').strip().casefold()

    def _csv_records(self):
        if not self.csv_path.exists(): return []
        with self.csv_path.open(encoding='utf-8-sig', newline='') as stream:
            return _in_scope([
                dict(source=self._subject(r['学科名称']), question=r['问题'].strip(), answer=r['答案'].strip())
                for r in csv.DictReader(stream) if r.get('问题', '').strip() and r.get('答案', '').strip()
            ])

    def fetch_faq(self):
        with self.lock:
            if self.conn is not None and self.cursor is not None:
                try:
                    self.conn.ping(reconnect=True)
                    self.cursor.execute('SELECT subject_name,question,answer FROM jpkb')
                    records = _in_scope([
                        dict(source=self._subject(s), question=q.strip(), answer=a.strip())
                        for s,q,a in self.cursor.fetchall() if q and a
                    ])
                    if records:
                        self.faq_source = "MySQL"
                        return records
                    logger.warning('MySQL FAQ 中没有当前学科的数据，回退本地 CSV')
                except Exception:
                    logger.warning('MySQL FAQ 读取失败，使用本地 CSV', exc_info=True)
                    self._shutdown_connection()
            records = self._csv_records()
            self.faq_source = "CSV"
            return records

    def fetch_questions(self):
        return [(r['question'],) for r in self.fetch_faq()]

    def answer(self, question):
        matches = [r['answer'] for r in self.fetch_faq() if r['question'] == question.strip()]
        return matches[0] if matches else None

    def create_table(self):
        with self.lock:
            if self.cursor is None: return
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS jpkb(
                id INT AUTO_INCREMENT PRIMARY KEY,subject_name VARCHAR(100),
                question TEXT,answer TEXT) CHARACTER SET utf8mb4''')

    def insert_data(self):
        with self.lock:
            if self.cursor is None: return
            records = self._csv_records()
            # 保持操作幂等，避免每次运行都重复插入同一份 CSV。
            for r in records:
                self.cursor.execute('SELECT id FROM jpkb WHERE subject_name=%s AND question=%s LIMIT 1',
                                    (r['source'],r['question']))
                if not self.cursor.fetchone():
                    self.cursor.execute('INSERT INTO jpkb(subject_name,question,answer) VALUES(%s,%s,%s)',
                                        (r['source'],r['question'],r['answer']))

    def close(self):
        with self.lock:
            self._shutdown_connection()
