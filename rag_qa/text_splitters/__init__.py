"""Text splitters with lazy optional semantic model support."""
from .chinese_recursive_splitter import ChineseRecursiveTextSplitter as ChineseRecursiveTextSplitter
__all__ = ['ChineseRecursiveTextSplitter', 'AliTextSplitter']
def __getattr__(name):
    if name == 'AliTextSplitter':
        from .semantic_splitter import AliTextSplitter
        return AliTextSplitter
    raise AttributeError(name)
