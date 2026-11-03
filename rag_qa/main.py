"""构建知识索引，或启动共用的交互式问答界面。"""
import argparse
from pathlib import Path


def main(query_mode=True, directory_path=None):
    if query_mode:
        from cli import main as chat
        return chat()
    from base.config import config
    from rag_qa.core.document_processor import process_documents
    from rag_qa.core.vector_store import VectorStore
    root=Path(directory_path or config.DATA_DIR)
    if not root.is_dir(): raise ValueError('资料目录不存在：'+str(root))
    store=VectorStore(load_reranker=False)
    total=0
    try:
        for source in config.VALID_SOURCES:
            directory=root/(source+'_data')
            if not directory.is_dir(): continue
            documents=process_documents(str(directory))
            store.replace_documents(documents)
            total+=len(documents)
        print('建库完成，共处理',total,'个知识块')
    finally:
        store.client.close()
    return total


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='学科知识问答：建库或交互提问')
    parser.add_argument('--data-processing',action='store_true')
    parser.add_argument('--data-dir')
    args=parser.parse_args()
    main(not args.data_processing,args.data_dir)
