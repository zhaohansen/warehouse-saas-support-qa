"""文本切分器集合：中文递归切分与语义切分。"""

from .edu_chinese_recursive_text_splitter import *

# 条件导入需要额外依赖的模块
try:
    from .edu_model_text_splitter import *
except ImportError:
    pass
