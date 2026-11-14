"""日志：统一格式与输出位置，供各模块复用。"""

import logging
import sys
from base.config import Config
from pathlib import Path
import tempfile

project_root = Path(__file__).resolve().parent.parent
log_file = project_root / Config.log_file
logger = logging.getLogger('warehouse-saas-support-qa')
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(console_formatter)
    logger.addHandler(error_handler)

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "warehouse-saas-support-qa"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(fallback_dir / "app.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)



class _TokenizerPerformanceFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        return not (message.startswith("You're using a XLMRobertaTokenizerFast tokenizer.")
                    and 'using the `__call__` method is faster' in message)


def filter_known_library_notices():
    """只隐藏已知的 AMD 实验性提示和 XLM-R 分词性能提示。"""
    import warnings
    for prefix in ('Flash Efficient attention', 'Mem Efficient attention'):
        warnings.filterwarnings(
            'ignore', message=rf'^{prefix} on Current AMD GPU is still experimental\.',
            category=UserWarning, module=r'^transformers\.models\.bert\.modeling_bert$')
    target = logging.getLogger('transformers.tokenization_utils_base')
    if not any(isinstance(item, _TokenizerPerformanceFilter) for item in target.filters):
        target.addFilter(_TokenizerPerformanceFilter())
