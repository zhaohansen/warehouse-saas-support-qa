"""按需共享 OCR 引擎，并对初始化和推理进行串行控制。"""
import logging
import threading

logger = logging.getLogger(__name__)
_engine_lock = threading.RLock()
_engines = {}


class _LockedOCR:
    def __init__(self, engine): self.engine = engine
    def __call__(self, *args, **kwargs):
        with _engine_lock: return self.engine(*args, **kwargs)


def get_ocr(use_cuda=True):
    with _engine_lock:
        if use_cuda in _engines: return _engines[use_cuda]
        engine = None
        try:
            from rapidocr_paddle import RapidOCR
            try:
                engine = RapidOCR(det_use_cuda=use_cuda,cls_use_cuda=use_cuda,rec_use_cuda=use_cuda)
            except Exception:
                if not use_cuda: raise
                logger.warning('Paddle OCR 显卡初始化失败，尝试 CPU')
                engine = RapidOCR(det_use_cuda=False,cls_use_cuda=False,rec_use_cuda=False)
        except (ImportError, RuntimeError, OSError, ValueError):
            logger.warning('Paddle OCR 不可用，尝试 ONNX Runtime')
        if engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                engine = RapidOCR()
            except ImportError as exc:
                raise RuntimeError('未安装可用的 OCR 引擎，请安装 rapidocr-onnxruntime') from exc
        wrapped = _LockedOCR(engine)
        _engines[use_cuda] = wrapped
        return wrapped
