"""幂等导入本地 FAQ CSV 到 MySQL 的 faq 表，供部署和演示重复执行。"""

import argparse
import sys

from base import Config
from mysql_qa.db.mysql_client import MySQLClient


def main(argv=None):
    parser = argparse.ArgumentParser(description='将 faq_knowledge_base.csv 导入 MySQL faq 表')
    parser.add_argument('--dry-run', action='store_true', help='只检查待导入记录，不写库')
    args = parser.parse_args(argv)

    client = MySQLClient()
    if client.conn is None or client.cursor is None:
        print(
            f"MySQL 连接失败（{Config.mysql_host}/{Config.mysql_db}）：未写入任何记录。"
            'FAQ 仍可使用本地 CSV 回退；请检查 MySQL 服务和 base/config.ini。',
            file=sys.stderr,
        )
        return 1

    try:
        if args.dry_run:
            pending, existing = client.pending_records()
            print(f'CSV 记录：{len(pending) + existing} 条；已存在：{existing} 条；待导入：{len(pending)} 条')
            for record in pending[:5]:
                print(f"  - [{record['source']}] {record['question']}")
            return 0

        client.create_table()
        before = client.source_counts()
        result = client.insert_data()
        after = client.source_counts()
        print(f"CSV 记录：{result['csv_rows']} 条；本次插入：{result['inserted']} 条；"
              f"已存在跳过：{result['skipped']} 条")
        print(f'资料分类记录（导入前）：{before}')
        print(f'资料分类记录（导入后）：{after}')
        return 0
    except Exception as exc:
        print(f'导入失败：{exc}', file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
