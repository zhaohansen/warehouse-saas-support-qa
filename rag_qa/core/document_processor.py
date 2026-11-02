"""文档处理：按扩展名选择加载器，切分为带父块信息的子块。"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Windows 控制台默认 GBK, 遇到生僻字符会抛 UnicodeEncodeError, 这里改为用替换字符输出
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from datetime import datetime
from rag_qa.edu_text_splitter import ChineseRecursiveTextSplitter
from rag_qa.edu_text_splitter import AliTextSplitter

from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, \
    OCRIMGLoader
from base.config import Config
from base.logger import logger

conf = Config()
document_loaders = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".png": OCRIMGLoader,
    ".md": UnstructuredMarkdownLoader
}


def load_documents_from_directory(directory_path):
    """
    从指定目录(含子目录)加载所有支持类型的文件, 为每个文档添加原数据
    :param directory_path: 目标文件夹的绝对路径.
    :return: 加载完成的文档列表, 每个元素为 LangChain Document对象, 含: page_content 和 metadata
    """
    documents = []
    supported_extensions = document_loaders.keys()
    source = os.path.basename(directory_path).replace('_data', '')
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()

            if file_extension in supported_extensions:
                try:
                    loader_class = document_loaders[file_extension]
                    if file_extension == '.txt':
                        loader = loader_class(file_path, encoding='utf-8')
                    else:
                        loader = loader_class(file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata["source"] = source
                        doc.metadata["file_path"] = file_path
                        doc.metadata["timestamp"] = datetime.now().isoformat()

                    documents.extend(loaded_docs)
                    logger.info(f'成功加载文件: {file_path}')
                except Exception as e:
                    logger.error(f'加载文件 {file_path} 失败: {str(e)}')
            else:
                logger.warning(f'不支持的文件类型: {file_path}')

    return documents


def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    """
    加载文档并进行分层气氛: 先切成大粒度父块, 再将父块切成小粒度子块, 子块关联父块元数据.
    :param directory_path:  文档目录路径
    :param parent_chunk_size: 父块切分长度(大粒度, 例如: 1200)
    :param child_chunk_size: 子块切分长度(小粒度, 例如: 300)
    :param chunk_overlap: 子块重叠长度, 确保上下文关联, 例如: 50
    :return:
        child_chunks: 切分后的子块列表(每个子块含父块关联信息)
    """

    documents = load_documents_from_directory(directory_path)
    logger.info(f"加载的文档数量: {len(documents)}")
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    child_chunks = []

    for i, doc in enumerate(documents):
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
        is_markdown = (file_extension == '.md')
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter

        logger.info(
            f'处理文档: {doc.metadata["file_path"]}, 使用切分器: {"Markdown" if is_markdown else "ChineseRecursive"}')
        parent_docs = parent_splitter_to_use.split_documents([doc])

        for j, parent_doc in enumerate(parent_docs):
            # 生成父块唯一id, 格式: doc_文档索引_parent_父块索引
            parent_id = f"doc_{i}_parent_{j}"
            parent_doc.metadata["parent_id"] = parent_id
            parent_doc.metadata["content"] = parent_doc.page_content
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])

            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata["parent_id"] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content

                # 生成子块唯一的ID, 格式: 父块ID_child_子块索引.
                sub_chunk.metadata["id"] = f'{parent_id}_child_{k}'
                child_chunks.append(sub_chunk)

    logger.info(f"切分完成的子块总数: {len(child_chunks)}")
    return child_chunks

if __name__ == '__main__':

    chunks = process_documents('../data/ai_data', conf.PARENT_CHUNK_SIZE, conf.CHILD_CHUNK_SIZE, conf.CHUNK_OVERLAP)
    print(f'切分完成的子块总数: {len(chunks)}')
    print(f'前1个子块详情: {chunks[0] if chunks else "未切分到任何子块"}')
