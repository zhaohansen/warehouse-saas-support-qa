"""持久化本地会话；每次操作使用各自的 SQLite 连接。"""
import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
import time

# WAL 模式下多个独立连接并发写入会触发 SQLITE_READONLY（attempt to write a
# readonly database），导致部分记录静默丢失。写操作在进程内串行化，读操作仍并发。
_WRITE_LOCK = threading.RLock()


class ConversationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, question TEXT NOT NULL,
                    answer TEXT NOT NULL, timestamp REAL NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    subject TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_history_session ON conversations(session_id, id);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            ''')
            columns = {row['name'] for row in db.execute('PRAGMA table_info(conversations)')}
            if 'subject' not in columns:
                db.execute("ALTER TABLE conversations ADD COLUMN subject TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, session_id, question, answer, sources=None, subject=None):
        if not session_id: return
        with _WRITE_LOCK, self.connect() as db:
            db.execute('''INSERT INTO conversations(session_id,question,answer,timestamp,sources,subject)
                          VALUES(?,?,?,?,?,?)''',
                       (session_id, question, answer, time.time(), json.dumps(sources or [], ensure_ascii=False),
                        subject or ''))

    def history(self, session_id, limit=200, subject=None):
        # 网页读取完整会话；模型上下文在指定学科时只读取同学科记录。
        clause = ' AND subject=?' if subject else ''
        parameters = (session_id, subject, max(1, min(int(limit), 500))) if subject else (
            session_id, max(1, min(int(limit), 500)))
        with self.connect() as db:
            rows = db.execute(
                'SELECT question,answer,sources,subject FROM conversations WHERE session_id=?' + clause +
                ' ORDER BY id DESC LIMIT ?', parameters).fetchall()
        return [{'question': row['question'], 'answer': row['answer'], 'sources': json.loads(row['sources']),
                 'subject': row['subject']} for row in reversed(rows)]

    def sessions(self, limit=50):
        with self.connect() as db:
            rows = db.execute('''SELECT session_id,COUNT(*) rounds,MAX(timestamp) updated_at,
                (SELECT question FROM conversations first WHERE first.session_id=c.session_id ORDER BY id LIMIT 1) title
                FROM conversations c GROUP BY session_id ORDER BY MAX(id) DESC LIMIT ?''',
                (max(1, min(int(limit), 200)),)).fetchall()
        return [dict(session_id=r['session_id'], rounds=r['rounds'],
                     title=r['title'].replace('\n', ' ')[:40],
                     updated_at=time.strftime('%Y-%m-%d %H:%M', time.localtime(r['updated_at']))) for r in rows]

    def clear(self, session_id):
        with _WRITE_LOCK, self.connect() as db:
            db.execute('DELETE FROM conversations WHERE session_id=?', (session_id,))
        return True

    def migrate_mysql(self, mysql_client):
        """一次性复制旧历史记录，不修改或删除 MySQL 中的原始记录。"""
        with _WRITE_LOCK, self.connect() as db:
            if db.execute("SELECT 1 FROM settings WHERE key='legacy_imported'").fetchone(): return
            if mysql_client.cursor is None: return
            with mysql_client.lock:
                mysql_client.conn.ping(reconnect=True)
                mysql_client.cursor.execute('SELECT session_id,question,answer,timestamp FROM conversations ORDER BY id')
                while True:
                    rows = mysql_client.cursor.fetchmany(500)
                    if not rows: break
                    for sid, question, answer, timestamp in rows:
                        stamp = timestamp.timestamp() if hasattr(timestamp, 'timestamp') else time.time()
                        db.execute('INSERT INTO conversations(session_id,question,answer,timestamp) VALUES(?,?,?,?)',
                                   (sid, question, answer, stamp))
            db.execute("INSERT INTO settings(key,value) VALUES('legacy_imported','1')")
