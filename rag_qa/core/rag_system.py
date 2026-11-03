"""Compatibility export: all entry points use the same RAG implementation."""
from rag_qa.core.rag_pipeline import RAGSystem as RAGSystem, _looks_like_subject_query
looks_like_subject_query = _looks_like_subject_query

__all__ = ["RAGSystem", "looks_like_subject_query"]
