"""向量库：BGE-M3 编码入库，Milvus 稠密 + 稀疏混合检索，CrossEncoder 重排。"""

import torch.cuda
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import hashlib
import sys, os

# 原来只有一句 `from document_processor import *`, 这是"隐式相对导入":
#   只有在 rag_qa/core 目录里直接运行脚本时才找得到模块.
#   一旦从项目根目录 import(例如 app.py / cli.py 这样用), 就会报
#   ModuleNotFoundError: No module named 'document_processor'.
# 这里先按原来的写法试, 失败再按"完整包路径"导一次, 两种运行方式都能用.
try:
    from document_processor import *
except ModuleNotFoundError:
    from rag_qa.core.document_processor import *

local_path = os.path.abspath(os.path.dirname(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

from base.config import Config
from base.logger import logger

conf = Config()


class VectorStore:
    def __init__(
            self,
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'使用设备: {self.device}')
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        self.reranker = CrossEncoder(reranker_path, device=self.device)
        m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,
            use_fp16=(self.device == 'cuda'),
            device=self.device)

        self.dense_dim = self.embedding_function.dim['dense']
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        self._create_or_load_collection()

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

    def add_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function(texts)
        data = []

        for i, doc in enumerate(documents):
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # 处理稀疏向量: BGE-M3返回的稀疏向量是矩阵, 需要转换为Milvus支持的字典格式.
            sparse_vector = {}
            row = embeddings['sparse'][[i]]
            indics = row.indices
            values = row.data
            for idx, value in zip(indics, values):

                sparse_vector[idx] = value

            # 组装单条插入数据 -> 字段要和Schema完全对应.
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })

        if data:
            # 使用upsert操作: 相同ID则更新, 不存在则插入(避免重复)
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(f'已插入/更新 {len(data)} 条数据到集合 {self.collection_name}')

    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
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
                logger.warning(f"忽略无效的来源过滤条件: {source_filter}")
                source_filter = None

        query_embeddings = self.embedding_function([query])
        dense_query_vector = query_embeddings["dense"][0]
        sparse_query_vector = {}
        row = query_embeddings["sparse"][[0]]
        indices = row.indices
        values = row.data
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field='dense_vector',
            param={'metric_type': 'IP', "params": {'nprobe': conf.NPROBE}},
            limit=k,
            expr=filter_expr
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field='sparse_vector',
            param={'metric_type': 'IP', "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 权重逻辑: 稠密向量侧重语义相似度, 稀疏向量侧重关键词匹配, 可根据业务调整.
        # 权重从配置读取，避免修改检索策略时必须改源码。
        ranker = WeightedRanker(conf.DENSE_WEIGHT, conf.SPARSE_WEIGHT)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]

        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]

        # 从子块中提取去重的父文档 -> 避免同一父块的多个子块重复返回.
        parent_docs = self._get_unique_parent_docs(sub_chunks)

        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]

        if parent_docs:
            # 构建'查询-文档'配对列表, 重排序模型需要改格式输入, 每个配对为 [query, doc_content]
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores = self.reranker.predict(pairs)
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]

        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        parent_contents = set()
        unique_docs = []
        for chunk in sub_chunks:
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            if parent_content and parent_content not in parent_contents:
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                parent_contents.add(parent_content)
        return unique_docs

    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )

if __name__ == '__main__':
    vector_store = VectorStore()
    directory_path = os.path.join(rag_qa_path, 'data', 'ai_data')
    documents = process_documents(directory_path)
    vector_store.add_documents(documents)
    query = '你认识夯哥吗?'
    results = vector_store.hybrid_search_with_rerank(query, source_filter='ai')
    print(f'results: {results}')
    print(f'results数量: {len(results)}')
