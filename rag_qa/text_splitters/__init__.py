"""文本切分器，按需加载可选的语义模型。"""
from .chinese_recursive_splitter import ChineseRecursiveTextSplitter as ChineseRecursiveTextSplitter
__all__ = ['ChineseRecursiveTextSplitter', 'AliTextSplitter']
def __getattr__(name):
    if name == 'AliTextSplitter':
        from .semantic_splitter import AliTextSplitter
        return AliTextSplitter
    raise AttributeError(name)
