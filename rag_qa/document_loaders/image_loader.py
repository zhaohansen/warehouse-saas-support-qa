"""图片加载器：OCR 识别图片中的文字。"""

from typing import Iterator
from rag_qa.document_loaders.ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class OCRIMGLoader(BaseLoader):
    """把一张图片里的文字 OCR 成单个 Document。"""

    def __init__(self, img_path: str) -> None:
        """使用图片路径初始化加载器。

        参数：
            img_path: 待识别的图片路径。
        """
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        """识别整张图片，产出包含文字与来源路径的单个 Document。"""
        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        resp = ""
        ocr = get_ocr()
        result, _ = ocr(self.img_path)
        if result:
            ocr_result = [line[1] for line in result]
            resp += "\n".join(ocr_result)
        return resp

if __name__ == '__main__':
    img_loader = OCRIMGLoader(img_path='sample/mysql_index_diagram.png')
    print(img_loader.load())
