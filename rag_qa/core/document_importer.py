"""使用共用的文档处理流程，将文件导入现有知识库。"""
import hashlib
import json
import time
import os
import re
from pathlib import Path
import shutil
import threading
import zipfile


class ImportValidationError(ValueError):
    pass


class DocumentImporter:
    MAX_BYTES = 20 * 1024 * 1024
    ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".txt"}

    def __init__(self, data_dir, valid_sources, vector_store, processor, batch_size=32):
        self.data_dir = Path(data_dir).resolve()
        self.valid_sources = set(valid_sources)
        self.vector_store = vector_store
        self.processor = processor
        self.batch_size = max(1, int(batch_size))
        self._lock = threading.Lock()

    def validate(self, source, filename):
        if source not in self.valid_sources or any(c in source for c in ('/', '\\')) or source in ('.', '..'):
            raise ImportValidationError("请选择有效的学科")
        if not filename or '/' in filename or '\\' in filename or '\x00' in filename:
            raise ImportValidationError("文件名无效")
        extension = Path(filename).suffix.lower()
        if extension == '.ppt':
            raise ImportValidationError("旧版 .ppt 不受支持，请先另存为 .pptx")
        if extension not in self.ALLOWED_EXTENSIONS:
            raise ImportValidationError("仅支持 PDF、PPTX 和 UTF-8 TXT")
        return extension

    def _validate_content(self, path, extension):
        size = path.stat().st_size
        if not size or size > self.MAX_BYTES:
            raise ImportValidationError("文件必须非空且不超过 20 MB")
        if extension == '.pdf':
            with path.open('rb') as stream:
                if b'%PDF-' not in stream.read(1024):
                    raise ImportValidationError("文件不是有效的 PDF")
        elif extension == '.pptx':
            try:
                with zipfile.ZipFile(path) as archive:
                    entries = archive.infolist()
                    if 'ppt/presentation.xml' not in archive.namelist():
                        raise ImportValidationError("文件不是有效的 PPTX")
                    if len(entries) > 10000 or sum(item.file_size for item in entries) > 200 * 1024 * 1024:
                        raise ImportValidationError("PPTX 解压后的内容过大，请拆分后导入")
            except zipfile.BadZipFile as exc:
                raise ImportValidationError("文件不是有效的 PPTX") from exc

    def import_file(self, upload_path, source, filename):
        extension = self.validate(source, filename)
        upload_path = Path(upload_path)
        self._validate_content(upload_path, extension)
        digest = hashlib.sha256(upload_path.read_bytes()).hexdigest()
        directory = (self.data_dir / (source + '_data') / 'uploads').resolve()
        if not directory.is_relative_to(self.data_dir):
            raise ImportValidationError("学科目录无效")
        destination = directory / (digest + extension)
        with self._lock:
            metadata_path = destination.with_name('.' + destination.name + '.json')
            if destination.exists():
                record = self._read_metadata(metadata_path)
                return {'status': 'already_imported', 'source': source, 'filename': filename,
                        'chunks': record.get('chunks', 0), 'warnings': record.get('warnings', [])}
            processed = list(self.processor(upload_path, source))
            warnings = list(dict.fromkeys(
                warning for doc in processed
                for warning in doc.metadata.get("warnings", [])
                if isinstance(warning, str)
            ))
            documents = [doc for doc in processed if doc.page_content.strip()]
            if not documents:
                raise ImportValidationError(
                    "未提取到文字，请确认文件包含可读取或可识别的内容；"
                    "若 PPT 含 WMF/EMF 等矢量图片，请先转为 PNG/JPG，或另存为 PDF 后重试。"
                )
            for doc in documents:
                doc.metadata.update(source=source, file_path=str(destination), filename=filename)
            directory.mkdir(parents=True, exist_ok=True)
            pending = directory / ('.' + digest + '.pending')
            rollback_documents = []
            try:
                # 先复制临时文件，全部批次完成索引后再正式保存。
                shutil.copyfile(upload_path, pending)
                if hasattr(self.vector_store, 'replace_documents'):
                    if hasattr(self.vector_store, 'existing_document_ids'):
                        from rag_qa.core.vector_store import document_vector_id
                        existing_ids = self.vector_store.existing_document_ids(documents)
                        rollback_documents = [doc for doc in documents
                                              if document_vector_id(doc) not in existing_ids]
                    else:
                        rollback_documents = list(documents)
                    self.vector_store.replace_documents(documents)
                else:
                    rollback_documents = list(documents)
                    for offset in range(0, len(documents), self.batch_size):
                        self.vector_store.add_documents(documents[offset:offset + self.batch_size])
                record = dict(filename=filename, source=source, chunks=len(documents), warnings=warnings,
                              size=upload_path.stat().st_size, imported_at=time.time())
                metadata_path.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
                os.replace(pending, destination)
            except Exception:
                # 即使回滚时无法连接 Milvus，稳定的 ID 也有助于安全重试。
                try:
                    if rollback_documents:
                        self.vector_store.delete_documents(rollback_documents)
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception('Failed to roll back document import')
                metadata_path.unlink(missing_ok=True)
                raise
            finally:
                pending.unlink(missing_ok=True)
            return {'status': 'imported', 'source': source, 'filename': filename, 'chunks': len(documents),
                    'warnings': warnings}

    @staticmethod
    def _read_metadata(path):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            return record if isinstance(record, dict) else {}
        except (OSError, ValueError):
            return {}

    def delete_file(self, source, document_id):
        """删除已导入文件及其向量；向量清理失败时保留文件以便重试。"""
        if (source not in self.valid_sources or any(c in source for c in ('/', '\\'))
                or source in ('.', '..')):
            raise ImportValidationError('请选择有效学科')
        if not isinstance(document_id, str) or not re.fullmatch(r'[0-9a-f]{64}\.(?:pdf|pptx|txt)', document_id):
            raise ImportValidationError('资料标识无效')
        directory = (self.data_dir / (source + '_data') / 'uploads').resolve()
        if not directory.is_relative_to(self.data_dir):
            raise ImportValidationError('学科目录无效')
        path = directory / document_id
        if path.is_symlink() or path.resolve().parent != directory:
            raise ImportValidationError('资料路径无效')
        with self._lock:
            if not path.is_file():
                raise FileNotFoundError('资料不存在或已删除')
            self.vector_store.delete_file_vectors(source, str(path))
            path.unlink()
            path.with_name('.' + path.name + '.json').unlink(missing_ok=True)
        return {'status': 'deleted'}

    def list_documents(self, source=None):
        if source and source not in self.valid_sources:
            raise ImportValidationError('请选择有效学科')
        documents = []
        for subject in sorted(self.valid_sources if not source else {source}):
            directory = self.data_dir / (subject + '_data') / 'uploads'
            if not directory.is_dir(): continue
            for path in directory.iterdir():
                if path.name.startswith(
                        '.') or path.suffix.lower() not in self.ALLOWED_EXTENSIONS or not path.is_file(): continue
                record = self._read_metadata(path.with_name('.' + path.name + '.json'))
                documents.append(dict(id=path.name, filename=record.get('filename', path.name), source=subject,
                                      size=path.stat().st_size, chunks=record.get('chunks'),
                                      imported_at=record.get('imported_at', path.stat().st_mtime),
                                      warnings=record.get('warnings', [])))
        return sorted(documents, key=lambda row: row['imported_at'], reverse=True)[:200]
