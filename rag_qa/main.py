"""构建知识索引，或启动共用的交互式问答界面。"""
import argparse
from pathlib import Path


def main(query_mode=True, directory_path=None, demo_private_data=False):
    if query_mode:
        from cli import main as chat
        return chat()
    from base.config import config
    from rag_qa.core.document_importer import DocumentImporter
    from rag_qa.core.document_processor import process_document_file, process_documents
    from rag_qa.core.vector_store import VectorStore
    if demo_private_data and not config.PRIVATE_MODE:
        raise ValueError('请在 QA_PRIVATE_MODE=1 的私有工作空间中导入演示资料')
    root=Path(directory_path or config.DATA_DIR)
    if not demo_private_data and not root.is_dir():
        raise ValueError('资料目录不存在：'+str(root))
    store=VectorStore(load_reranker=False)
    total=0
    try:
        if demo_private_data:
            importer = DocumentImporter(root, config.VALID_SOURCES, store, process_document_file)
            for path in sorted(Path(__file__).with_name('demo_private').glob('*.txt')):
                result = importer.import_file(path, 'internal_docs', path.name)
                print(f"{path.name}: {result['status']} ({result['chunks']} 个知识块)")
                total += result['chunks'] if result['status'] == 'imported' else 0
            return total
        for source in config.VALID_SOURCES:
            directory=root/source
            if not directory.is_dir(): continue
            documents=process_documents(str(directory))
            store.replace_documents(documents)
            total+=len(documents)
        print('建库完成，共处理',total,'个知识块')
    finally:
        store.client.close()
    return total


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='仓配 SaaS 实施与售后知识问答：建库或交互提问')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--data-processing',action='store_true')
    mode.add_argument('--demo-private-data',action='store_true')
    parser.add_argument('--data-dir')
    args=parser.parse_args()
    if args.demo_private_data and args.data_dir:
        parser.error('--demo-private-data 使用私有工作空间的默认资料目录')
    main(not (args.data_processing or args.demo_private_data), args.data_dir, args.demo_private_data)
