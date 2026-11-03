"""Load only the requested document parser."""
from importlib import import_module
__all__ = ['OCRPDFLoader', 'OCRPPTLoader', 'OCRDOCLoader', 'OCRIMGLoader']
_MODULES = {'OCRPDFLoader':'pdf_loader','OCRPPTLoader':'ppt_loader','OCRDOCLoader':'docx_loader','OCRIMGLoader':'image_loader'}
def __getattr__(name):
    if name not in _MODULES: raise AttributeError(name)
    return getattr(import_module('.' + _MODULES[name], __name__), name)
