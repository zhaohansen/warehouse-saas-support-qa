import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'rag_qa/core/document_importer.py'
spec = importlib.util.spec_from_file_location('importer', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeVectorStore:
    def __init__(self, fail=False): self.calls, self.fail = [], fail
    def delete_file_vectors(self, source, path):
        self.calls.append((source, path))
        if self.fail: raise RuntimeError('向量库不可用')

class DeleteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'python_data' / 'uploads'
        self.directory.mkdir(parents=True)
        self.identifier = 'a' * 64 + '.txt'
        self.path = self.directory / self.identifier
        self.path.write_text('测试资料', encoding='utf-8')
        self.meta = self.directory / ('.' + self.identifier + '.json')
        self.meta.write_text(json.dumps({'filename':'学习资料.txt'}), encoding='utf-8')
        self.store = FakeVectorStore()
        self.importer = module.DocumentImporter(self.root, ['python'], self.store, None)

    def test_delete_scoped_file_and_vectors(self):
        self.assertEqual(self.importer.list_documents()[0]['id'], self.identifier)
        self.importer.delete_file('python', self.identifier)
        self.assertEqual(self.store.calls, [('python', str(self.path))])
        self.assertFalse(self.path.exists())
        self.assertFalse(self.meta.exists())

    def test_vector_failure_preserves_file_for_retry(self):
        self.store.fail = True
        with self.assertRaisesRegex(RuntimeError, '向量库不可用'):
            self.importer.delete_file('python', self.identifier)
        self.assertTrue(self.path.exists())
        self.assertTrue(self.meta.exists())

    def test_invalid_identifier_cannot_delete(self):
        with self.assertRaises(module.ImportValidationError):
            self.importer.delete_file('python', '../anything.txt')
        self.assertTrue(self.path.exists())
        self.assertEqual(self.store.calls, [])

    def test_not_found_does_not_delete_vectors(self):
        with self.assertRaises(FileNotFoundError):
            self.importer.delete_file('python', 'b' * 64 + '.txt')
        self.assertEqual(self.store.calls, [])

if __name__ == '__main__': unittest.main()
