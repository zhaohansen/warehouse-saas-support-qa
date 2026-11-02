"""OCR 封装：初始化识别器，统一输出文本与置信度。"""

from typing import TYPE_CHECKING


def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr
