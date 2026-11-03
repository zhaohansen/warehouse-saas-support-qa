"""PPTX loader: preserve slide references and isolate unsupported embedded images."""
import logging
from io import BytesIO
from typing import Iterator
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from rag_qa.document_loaders.ocr import get_ocr

logger = logging.getLogger(__name__)


class OCRPPTLoader(BaseLoader):
    def __init__(self, filepath):
        self.filepath = str(filepath)
        self.warnings = []

    def lazy_load(self) -> Iterator[Document]:
        presentation = Presentation(self.filepath)
        self.warnings = []
        ocr = None
        for number, slide in enumerate(presentation.slides,1):
            fragments, warnings = [], []
            def extract(shape):
                nonlocal ocr
                if shape.has_text_frame and shape.text.strip(): fragments.append(shape.text.strip())
                if shape.has_table:
                    for row in shape.table.rows:
                        fragments.append('\t'.join(cell.text.strip() for cell in row.cells))
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        with Image.open(BytesIO(shape.image.blob)) as image:
                            array = np.array(image.convert('RGB'))
                    except (OSError, ValueError) as exc:
                        warning = f'第 {number} 页图片「{shape.name}」无法解码，已跳过该图的文字识别；如需识别，请先将该图转为 PNG/JPG。'
                        warnings.append(warning)
                        logger.warning('%s 原因：%s',warning,exc)
                        return
                    if ocr is None: ocr = get_ocr()
                    result,_ = ocr(array)
                    fragments.extend(str(item[1]) for item in result or [])
                elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    for child in sorted(shape.shapes,key=lambda item:(item.top,item.left)): extract(child)
            for shape in sorted(slide.shapes,key=lambda item:(item.top,item.left)): extract(shape)
            self.warnings.extend(warnings)
            yield Document(page_content='\n'.join(fragments),metadata={'source':self.filepath,'slide':number,'warnings':warnings})

    def ppt2text(self, filepath=None):
        if filepath is not None: self.filepath = str(filepath)
        return '\n\n'.join(doc.page_content for doc in self.lazy_load())
