"""串行访问 MySQL 常见问答；启动或运行时失败可回退到 CSV。"""
import csv
from pathlib import Path
import threading
import pymysql
from base import Config
from base.logger import logger


def _in_scope(records):
    """只保留配置中的资料分类。"""
    allowed = {str(s).strip().casefold() for s in Config.VALID_SOURCES}
    return [r for r in records if str(r.get('source', '')).strip().casefold() in allowed]


class MySQLClient:
    def __init__(self):
        self.lock = threading.RLock()
        self.conn = self.cursor = None
        self.faq_source = None
        self._has_source_doc = None
        self.csv_path = Path(__file__).with_name('faq_knowledge_base.csv')
        try:
            self.conn = pymysql.connect(host=Config.mysql_host, user=Config.mysql_user,
                password=Config.mysql_password, database=Config.mysql_db, charset='utf8mb4',
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
        return str(value).strip().casefold()

    def _csv_records(self):
        if not self.csv_path.exists(): return []
        with self.csv_path.open(encoding='utf-8-sig', newline='') as stream:
            records = []
            for row in csv.DictReader(stream):
                question = (row.get('问题') or '').strip()
                answer = (row.get('答案') or '').strip()
                if not question or not answer: continue
                records.append(dict(
                    source=self._subject(row.get('资料分类', '')), question=question, answer=answer,
                    source_doc=(row.get('来源文档') or '').strip()))
            return _in_scope(records)

    def _detect_source_doc(self):
        """探测 faq 表是否已有 source_doc 列；探测失败不影响 MySQL 连接。"""
        if self._has_source_doc is None:
            try:
                self.cursor.execute(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s "
                    "AND TABLE_NAME='faq' AND COLUMN_NAME='source_doc'", (Config.mysql_db,))
                self._has_source_doc = bool(self.cursor.fetchone()[0])
            except Exception:
                logger.warning('检测 faq.source_doc 列失败，按旧表结构读取', exc_info=True)
                self._has_source_doc = False
        return self._has_source_doc

    def fetch_faq(self):
        with self.lock:
            if self.conn is not None and self.cursor is not None:
                try:
                    self.conn.ping(reconnect=True)
                    if self._detect_source_doc():
                        self.cursor.execute('SELECT source,question,answer,source_doc FROM faq')
                        records = _in_scope([
                            dict(source=self._subject(s), question=q.strip(), answer=a.strip(),
                                 source_doc=(d or '').strip())
                            for s,q,a,d in self.cursor.fetchall() if q and a
                        ])
                    else:
                        self.cursor.execute('SELECT source,question,answer FROM faq')
                        records = _in_scope([
                            dict(source=self._subject(s), question=q.strip(), answer=a.strip(), source_doc='')
                            for s,q,a in self.cursor.fetchall() if q and a
                        ])
                    if records:
                        self.faq_source = "MySQL"
                        return records
                    logger.warning('MySQL FAQ 中没有当前分类的数据，回退本地 CSV')
                except Exception:
                    logger.warning('MySQL FAQ 读取失败，使用本地 CSV', exc_info=True)
                    self._shutdown_connection()
                    self._has_source_doc = None
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
            if self.cursor is None:
                raise RuntimeError('MySQL 不可用，无法建表或补齐字段')
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS faq(
                id INT AUTO_INCREMENT PRIMARY KEY,source VARCHAR(100),
                question TEXT,answer TEXT,
                source_doc VARCHAR(512) DEFAULT '') CHARACTER SET utf8mb4''')
            self.cursor.execute(
                "SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s "
                "AND TABLE_NAME='faq' AND COLUMN_NAME='source_doc'", (Config.mysql_db,))
            if not self.cursor.fetchone()[0]:
                try:
                    self.cursor.execute("ALTER TABLE faq ADD COLUMN source_doc VARCHAR(512) DEFAULT ''")
                except pymysql.err.OperationalError as exc:
                    # 并发初始化时可能已被其他进程补列，1060 可安全忽略。
                    if not exc.args or exc.args[0] != 1060: raise
            self._has_source_doc = True

    def _existing(self, source, question):
        self.cursor.execute('SELECT id FROM faq WHERE source=%s AND question=%s LIMIT 1',
                            (source, question))
        return self.cursor.fetchone() is not None

    def insert_data(self):
        with self.lock:
            if self.cursor is None:
                raise RuntimeError('MySQL 不可用，无法导入 FAQ')
            records = self._csv_records()
            inserted = skipped = 0
            # 保持操作幂等：只插入不存在的记录，绝不覆盖已有行或来源不明的数据。
            for r in records:
                if self._existing(r['source'], r['question']):
                    skipped += 1
                    continue
                self.cursor.execute(
                    'INSERT INTO faq(source,question,answer,source_doc) VALUES(%s,%s,%s,%s)',
                    (r['source'], r['question'], r['answer'], r.get('source_doc', '')))
                inserted += 1
            return dict(csv_rows=len(records), inserted=inserted, skipped=skipped)

    def pending_records(self):
        """供 dry-run 使用：返回尚未入库的 CSV 记录与已存在数量。"""
        with self.lock:
            if self.cursor is None:
                raise RuntimeError('MySQL 不可用，无法检查待导入记录')
            records = self._csv_records()
            self.cursor.execute(
                "SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s "
                "AND TABLE_NAME='faq'", (Config.mysql_db,))
            if not self.cursor.fetchone()[0]:
                return records, 0
            pending = [r for r in records if not self._existing(r['source'], r['question'])]
            return pending, len(records) - len(pending)

    def source_counts(self):
        with self.lock:
            if self.cursor is None:
                raise RuntimeError('MySQL 不可用，无法统计资料分类记录')
            sources = list(Config.VALID_SOURCES)
            placeholders = ','.join(['%s'] * len(sources))
            self.cursor.execute(
                f'SELECT source,COUNT(*) FROM faq WHERE source IN ({placeholders}) '
                'GROUP BY source', tuple(sources))
            return {str(name): int(count) for name, count in self.cursor.fetchall()}

    def close(self):
        with self.lock:
            self._has_source_doc = None
            self._shutdown_connection()
