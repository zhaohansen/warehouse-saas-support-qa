"""PDF loader: retain page references and OCR large-image pages in displayed orientation."""
import logging
import math
from typing import Iterator
import cv2
import fitz
import numpy as np
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from rag_qa.document_loaders.ocr import get_ocr

PDF_OCR_THRESHOLD = (0.6, 0.6)
logger = logging.getLogger(__name__)


class OCRPDFLoader(BaseLoader):
    def __init__(self, file_path): self.file_path = str(file_path)

    def lazy_load(self) -> Iterator[Document]:
        ocr = None
        with fitz.open(self.file_path) as document:
            if document.needs_pass: raise ValueError('PDF 已加密，请先解密后再导入')
            for number, page in enumerate(document, 1):
                text = page.get_text('text')
                warnings = []
                # Image bounding boxes use unrotated page coordinates.
                width, height = page.cropbox.width, page.cropbox.height
                large_image = any((item['bbox'][2]-item['bbox'][0]) / max(width,1) >= PDF_OCR_THRESHOLD[0]
                                  and (item['bbox'][3]-item['bbox'][1]) / max(height,1) >= PDF_OCR_THRESHOLD[1]
                                  for item in page.get_image_info())
                if large_image:
                    try:
                        # Rendering the page respects its rotation and each embedded image transform.
                        # Cap pixel count before allocation for unusually large page dimensions.
                        scale = min(150/72, math.sqrt(12_000_000/max(page.rect.width*page.rect.height,1)))
                        pix = page.get_pixmap(matrix=fitz.Matrix(scale,scale), alpha=False, colorspace=fitz.csRGB)
                        array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height,pix.width,3)
                    except (OSError, ValueError, RuntimeError) as exc:
                        warnings.append(f'第 {number} 页无法渲染图片，已保留可读取的文字。')
                        logger.warning('PDF page %s render failed: %s', number, exc)
                    else:
                        if ocr is None: ocr = get_ocr()
                        result, _ = ocr(array)
                        native = ''.join(text.split()).casefold()
                        extra = []
                        for item in result or []:
                            line = str(item[1]).strip()
                            normalized = ''.join(line.split()).casefold()
                            if normalized and normalized not in native:
                                extra.append(line)
                                native += normalized
                        text += '\n' + '\n'.join(extra)
                yield Document(page_content=text.strip(),metadata={'source':self.file_path,'page':number,'warnings':warnings})

    def pdf2text(self): return '\n\n'.join(doc.page_content for doc in self.lazy_load())

    def rotate_img(self, img, angle):
        h,w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
        # Rounding errors near right angles must not add or remove a pixel.
        new_w = math.ceil(round(h*abs(matrix[0,1])+w*abs(matrix[0,0]),8))
        new_h = math.ceil(round(h*abs(matrix[0,0])+w*abs(matrix[0,1]),8))
        matrix[0,2] += (new_w-w)/2
        matrix[1,2] += (new_h-h)/2
        return cv2.warpAffine(img,matrix,(new_w,new_h),borderValue=(255,255,255))
