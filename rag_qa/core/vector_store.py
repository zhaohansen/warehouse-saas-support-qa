"""向量库：BGE-M3 编码入库，Milvus 稠密 + 稀疏混合检索，CrossEncoder 重排。"""

import torch.cuda
import numpy as np
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import hashlib
import json
import threading
from functools import wraps
import sys, os

local_path = os.path.abspath(os.path.dirname(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

from base.config import Config
from base.logger import logger

conf = Config()


def _resolve_device(device_name):
    """配置为 cuda 但当前环境不可用时回退 CPU, 避免启动直接失败."""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(f"配置设备 {device_name} 不可用, 已回退到 CPU")
        return "cpu"
    return device_name


def _as_float32(vector):
    """
    稠密向量统一转 float32.

    use_fp16=true 时 BGE-M3 输出的 numpy 数组是 float16, 而 collection 的
    dense_vector 字段类型是 FLOAT_VECTOR(float32). insert 时 pymilvus 会替我们转换,
    search 时却按数组真实的 dtype 上报向量类型, 于是 Milvus 报:
        vector type must be the same, field dense_vector - type VECTOR_FLOAT,
        search info type VECTOR_FLOAT16
    这里显式转换, 不依赖用例路径上的隐式行为.
    """
    return np.asarray(vector, dtype=np.float32)


def resolve_model_path(configured, fallback_name):
    """相对路径以项目根目录为基准，模型目录不存在时明确报错。"""
    if os.path.isabs(configured):
        candidates = [os.path.abspath(configured)]
    else:
        candidates = [os.path.abspath(os.path.join(project_root, configured))]
        if os.path.basename(os.path.normpath(configured)) == fallback_name:
            fallback = os.path.abspath(os.path.join(rag_qa_path, 'models', fallback_name))
            if fallback not in candidates:
                candidates.append(fallback)

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"模型文件夹不存在，已检查：{', '.join(candidates)}")


def document_vector_id(doc):
    identity = [doc.metadata.get("source"), doc.metadata.get("file_path"),
                doc.metadata.get("page"), doc.metadata.get("slide"),
                doc.metadata.get("parent_content"), doc.page_content]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()


def serialized_model(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._model_lock:
            return method(self, *args, **kwargs)

    return guarded


class VectorStore:
    def __init__(
            self,
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME,
            load_reranker=True
    ):
        self._model_lock = threading.RLock()
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger
        self.device = _resolve_device(conf.EMBEDDING_DEVICE)
        self.reranker_device = _resolve_device(conf.RERANKER_DEVICE)
        self.logger.info(
            f'Embedding 使用设备: {self.device}, Reranker 使用设备: {self.reranker_device}'
        )
        self.reranker_path = resolve_model_path(conf.RERANKER_MODEL_PATH, 'bge-reranker-large')
        self.reranker = CrossEncoder(self.reranker_path, device=self.reranker_device) if load_reranker else None
        m3_path = resolve_model_path(conf.EMBEDDING_MODEL_PATH, 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,
            use_fp16=(conf.EMBEDDING_USE_FP16 and self.device != 'cpu'),
            device=self.device, devices=[self.device])

        self.dense_dim = self.embedding_function.dim['dense']
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        try:
            self._create_or_load_collection()
        except Exception:
            close = getattr(self.client, 'close', None)
            if close:
                try:
                    close()
                except Exception:
                    logger.debug('Milvus 初始化失败后关闭客户端失败', exc_info=True)
            raise

    def _create_or_load_collection(self):
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            logger.info(f"已创建集合 {self.collection_name}")

        else:
            logger.info(f"已加载集合 {self.collection_name}")
        self.client.load_collection(self.collection_name)

    @serialized_model
    def add_documents(self, documents):
        batch_size = max(1, conf.BATCH_SIZE)
        for offset in range(0, len(documents), batch_size):
            self._add_document_batch(documents[offset:offset + batch_size])

    def _add_document_batch(self, documents):
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function(texts)
        data = []

        for i, doc in enumerate(documents):
            text_hash = document_vector_id(doc)
            # 处理稀疏向量: BGE-M3返回的稀疏向量是矩阵, 需要转换为Milvus支持的字典格式.
            sparse_vector = {}
            row = embeddings['sparse'][[i]]
            indics = row.indices
            values = row.data
            for idx, value in zip(indics, values):
                sparse_vector[int(idx)] = float(value)

            # 组装单条插入数据 -> 字段要和Schema完全对应.
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": _as_float32(embeddings["dense"][i]),
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown"),
                "file_path": doc.metadata.get("file_path", ""),
                "filename": doc.metadata.get("filename") or os.path.basename(doc.metadata.get("file_path", "")),
                "page": doc.metadata.get("page"),
                "slide": doc.metadata.get("slide")
            })

        if data:
            # 使用upsert操作: 相同ID则更新, 不存在则插入(避免重复)
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(f'已插入/更新 {len(data)} 条数据到集合 {self.collection_name}')

    def existing_document_ids(self, documents):
        """返回写入前已经存在的稳定 ID，供导入失败时安全回滚。"""
        ids = [document_vector_id(doc) for doc in documents]
        existing = set()
        for offset in range(0, len(ids), 100):
            rows = self.client.get(collection_name=self.collection_name, ids=ids[offset:offset + 100],
                                   output_fields=['id']) or []
            existing.update(row['id'] for row in rows)
        return existing

    @serialized_model
    def replace_documents(self, documents):
        """写入新版 ID 的文档块，并清理同一文件中不再使用的记录。"""
        if not documents: return
        old_ids = set()
        scopes = {(doc.metadata.get('source'), doc.metadata.get('file_path')) for doc in documents}
        for source, path in scopes:
            if not source or not path: continue
            expression = 'source == ' + json.dumps(source, ensure_ascii=False) + ' and file_path == ' + json.dumps(path,
                                                                                                                   ensure_ascii=False)
            iterator = self.client.query_iterator(collection_name=self.collection_name, filter=expression,
                                                  output_fields=['id'], batch_size=500, consistency_level='Strong')
            try:
                while True:
                    rows = iterator.next()
                    if not rows: break
                    old_ids.update(row['id'] for row in rows)
            finally:
                iterator.close()
        new_ids = {document_vector_id(doc) for doc in documents}
        try:
            self.add_documents(documents)
        except Exception:
            created = list(new_ids - old_ids)
            for offset in range(0, len(created), 100):
                self.client.delete(collection_name=self.collection_name, ids=created[offset:offset + 100])
            raise
        obsolete = list(old_ids - new_ids)
        try:
            for offset in range(0, len(obsolete), 100):
                self.client.delete(collection_name=self.collection_name, ids=obsolete[offset:offset + 100])
        except Exception:
            # 新数据已经可用；旧记录清理失败时可稍后重试。
            logger.warning('新索引已写入，部分旧索引清理失败，可重试建库', exc_info=True)

    def delete_documents(self, documents):
        ids = [document_vector_id(doc) for doc in documents]
        for offset in range(0, len(ids), 100):
            self.client.delete(collection_name=self.collection_name, ids=ids[offset:offset + 100])

    @serialized_model
    def delete_file_vectors(self, source, file_path):
        """按学科和文件路径删除该文件的向量记录。"""
        expression = ('source == ' + json.dumps(source, ensure_ascii=False)
                      + ' and file_path == ' + json.dumps(file_path, ensure_ascii=False))
        self.client.delete(collection_name=self.collection_name, filter=expression)

    @serialized_model
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None, rerank_query=None):
        """
        该函数用于执行: 混合检索(稠密 + 稀疏向量) + 结果重排序, 返回精准父文档
        :param query: 用户查询文本, 例如: 'AI学科的课程内容是什么'
        :param k: 混合检索返回的TopK子块数量
        :param source_filter: 学科过滤条件, 例如: ai, 仅检索该学科的文档, 默认None不过滤.
        :return: 重排序后的Top-M个父文档列表(M是从conf中获取的)
        """
        # 来源过滤统一校验，避免未经校验的字符串拼进 Milvus 表达式。
        if source_filter:
            source_filter = str(source_filter).strip()
            if source_filter not in conf.VALID_SOURCES:
                raise ValueError("无效的学科过滤条件")

        # 候选召回数量与最终上下文数量分别控制。
        # 多个命中的子块可能都来自同一张相关性较弱的幻灯片。
        recall_k = max(k, conf.RECALL_CANDIDATES)
        query_embeddings = self.embedding_function([query])
        dense_query_vector = _as_float32(query_embeddings["dense"][0])
        sparse_query_vector = {}
        row = query_embeddings["sparse"][[0]]
        indices = row.indices
        values = row.data
        for idx, value in zip(indices, values):
            sparse_query_vector[int(idx)] = float(value)

        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field='dense_vector',
            param={'metric_type': 'IP', "params": {'nprobe': conf.NPROBE}},
            limit=recall_k,
            expr=filter_expr
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field='sparse_vector',
            param={'metric_type': 'IP', "params": {}},
            limit=recall_k,
            expr=filter_expr
        )

        # 权重逻辑: 稠密向量侧重语义相似度, 稀疏向量侧重关键词匹配, 可根据业务调整.
        # 权重从配置读取，避免修改检索策略时必须改源码。
        ranker = WeightedRanker(conf.DENSE_WEIGHT, conf.SPARSE_WEIGHT)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=recall_k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp", "filename", "file_path",
                           "page", "slide"],
            consistency_level='Strong'
        )[0]

        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]

        # 从子块中提取去重的父文档 -> 避免同一父块的多个子块重复返回.
        parent_docs = self._get_unique_parent_docs(sub_chunks)

        return self.rerank_documents(rerank_query or query, parent_docs)

    @serialized_model
    def rerank_documents(self, query, documents):
        if not documents: return []
        if self.reranker is None:
            self.reranker = CrossEncoder(self.reranker_path, device=self.reranker_device)
        pairs = [[query, doc.page_content] for doc in documents]
        scores = np.atleast_1d(self.reranker.predict(pairs, activation_fct=torch.nn.Sigmoid()))
        ranked = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        result = []
        for score, doc in ranked:
            if not np.isfinite(score) or float(score) < conf.MIN_SCORE: continue
            doc.metadata['score'] = float(score)
            result.append(doc)
        return result[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        parent_contents = set()
        unique_docs = []
        for chunk in sub_chunks:
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            key = (chunk.metadata.get('source'), chunk.metadata.get('file_path'),
                   chunk.metadata.get('page'), chunk.metadata.get('slide'),
                   chunk.metadata.get('parent_id'), parent_content)
            if parent_content and key not in parent_contents:
                unique_docs.append(Document(page_content=parent_content, metadata=dict(chunk.metadata)))
                parent_contents.add(key)
        return unique_docs

    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp"),
                "file_path": hit.get("file_path"),
                "filename": hit.get("filename"),
                "page": hit.get("page"),
                "slide": hit.get("slide")
            }
        )


if __name__ == '__main__':
    from rag_qa.core.document_processor import process_documents

    # 按配置中的学科依次读取资料，并写入当前 Milvus 集合。
    vector_store = VectorStore(load_reranker=False)
    total = 0
    try:
        for source in conf.VALID_SOURCES:
            directory_path = os.path.join(conf.DATA_DIR, f'{source}_data')
            if not os.path.isdir(directory_path):
                print(f'资料目录不存在，跳过：{directory_path}')
                continue

            print(f'正在处理 {source} 资料：{directory_path}')
            documents = process_documents(directory_path)
            vector_store.replace_documents(documents)
            total += len(documents)
            print(f'{source} 处理完成，共 {len(documents)} 个文档块')

        print(f'建库完成，共处理 {total} 个文档块')
    finally:
        vector_store.client.close()
