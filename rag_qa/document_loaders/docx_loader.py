"""解析 Word 文档，保留段落和表格顺序，并单独处理不支持的图片。"""
import logging
from io import BytesIO
import numpy as np
from PIL import Image
from docx import Document as WordDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from rag_qa.document_loaders.ocr import get_ocr

logger=logging.getLogger(__name__)


class OCRDOCLoader(BaseLoader):
    def __init__(self,filepath):self.filepath=str(filepath);self.warnings=[]
    def lazy_load(self):
        text=self.doc2text(self.filepath)
        yield Document(page_content=text,metadata={'source':self.filepath,'warnings':list(self.warnings)})
    def doc2text(self,filepath):
        document=WordDocument(filepath);self.warnings=[];ocr=None
        def paragraph_text(paragraph):
            nonlocal ocr
            parts=[paragraph.text.strip()] if paragraph.text.strip() else []
            for reference in paragraph._element.xpath('.//a:blip/@r:embed'):
                part=paragraph.part.related_parts.get(reference)
                if part is None:continue
                try:
                    with Image.open(BytesIO(part.blob)) as image:array=np.array(image.convert('RGB'))
                except (OSError,ValueError) as exc:
                    warning='Word 中一张图片无法解码，已跳过该图的文字识别；如需识别，请先将该图转为 PNG/JPG。'
                    self.warnings.append(warning);logger.warning('%s 原因：%s',warning,exc);continue
                if ocr is None:ocr=get_ocr()
                result,_=ocr(array);parts.extend(str(item[1]) for item in result or [])
            return '\n'.join(parts)
        def extract(parent,element):
            parts=[]
            for child in element.iterchildren():
                if isinstance(child,CT_P):parts.append(paragraph_text(Paragraph(child,parent)))
                elif isinstance(child,CT_Tbl):
                    table=Table(child,parent);seen=set()
                    for row in table.rows:
                        cells=[]
                        for cell in row.cells:
                            if cell._tc in seen:continue
                            seen.add(cell._tc);cells.append(extract(cell,cell._tc))
                        parts.append('\t'.join(cells))
            return '\n'.join(part for part in parts if part)
        return extract(document,document.element.body)
