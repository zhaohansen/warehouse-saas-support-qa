from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rag_qa.core.document_importer import DocumentImporter, ImportValidationError
from rag_qa.core.document_processor import load_document_file, process_document_file


class RecordingStore:
    def __init__(self):
        self.chunks = []

    def add_documents(self, chunks):
        self.chunks.extend(chunks)


class DocumentImportTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.upload = self.root / 'upload.txt'
        self.store = RecordingStore()
        self.importer = DocumentImporter(self.root, ['product_docs'], self.store, process_document_file)

    def test_upload_without_title_uses_original_filename(self):
        self.upload.write_text('这是库存查询说明。', encoding='utf-8')
        self.importer.import_file(self.upload, 'product_docs', '库存手册.txt')
        self.assertEqual(self.store.chunks[0].metadata['doc_title'], '库存手册.txt')

    def test_header_and_upload_metadata_remain_consistent_with_index(self):
        self.upload.write_text('【文档元数据】\n文档标题：库存手册\n产品：旺店通ERP\n平台：通用\n'
                               '【正文】\n这是库存查询说明。', encoding='utf-8')
        self.importer.import_file(self.upload, 'product_docs', '库存手册.txt', {'platform': '旺店通开放平台'})
        chunk = self.store.chunks[0]
        listed = self.importer.list_documents()[0]
        reloaded = load_document_file(chunk.metadata['file_path'], 'product_docs')[0]
        self.assertEqual(chunk.metadata['doc_title'], '库存手册')
        self.assertEqual(listed['product'], '旺店通ERP')
        self.assertEqual(listed['platform'], '旺店通开放平台')
        self.assertEqual(reloaded.metadata['platform'], listed['platform'])

    def test_metadata_only_file_is_not_indexed(self):
        self.upload.write_text('【文档元数据】\n文档标题：库存手册\n【正文】\n', encoding='utf-8')
        with self.assertRaises(ImportValidationError):
            self.importer.import_file(self.upload, 'product_docs', '库存手册.txt')
        self.assertEqual(self.store.chunks, [])
        self.assertEqual(self.importer.list_documents(), [])
