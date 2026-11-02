"""RAG 通道入口：按需导出，避免 import 时加载模型与向量库。"""

# 注意: 这里不做重量级导入(向量库/大模型), 否则只用到文档加载器时也会被迫加载模型.
import os
import sys

current_dir = os.path.abspath(__file__)
rag_qa_path = os.path.dirname(current_dir)

if rag_qa_path not in sys.path:
    sys.path.insert(0, rag_qa_path)

__all__ = ["RAGPrompts", "VectorStore", "RAGSystem"]


def __getattr__(name):
    if name == "RAGPrompts":
        from rag_qa.core.prompts import RAGPrompts
        return RAGPrompts
    if name == "VectorStore":
        from rag_qa.core.vector_store import VectorStore
        return VectorStore
    if name == "RAGSystem":
        from rag_qa.core.rag_pipeline import RAGSystem
        return RAGSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
