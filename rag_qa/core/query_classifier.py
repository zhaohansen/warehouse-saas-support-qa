"""BERT 问题二分类：训练、评估和预测通用知识 / 学科资料标签。"""

import json
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

import torch
from base.logger import logger
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainerCallback, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class _ROCmTrainingLayerNorm(torch.nn.LayerNorm):
    def forward(self, inputs):
        result = torch.nn.functional.layer_norm(
            inputs.to('cpu'), self.normalized_shape,
            self.weight.to('cpu') if self.weight is not None else None,
            self.bias.to('cpu') if self.bias is not None else None,
            self.eps,
        )
        return result.to(inputs.device)


class _CheckFiniteGradients(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f'训练梯度出现无效数值，已停止训练：{name}')


class QueryClassifier:
    def __init__(self, model_path='../models/bert_query_classifier'):
        # 相对路径以当前源码目录为基准，避免从不同工作目录启动时加载失败。
        self.model_path = (
            os.path.normpath(os.path.join(current_dir, model_path))
            if not os.path.isabs(model_path) else model_path
        )
        self.bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {self.device}')
        # 标签的具体判定依据取决于训练数据的标注规则。
        self.label_map = {"通用知识": 0, "学科资料": 1}
        self.load_model()

    def load_model(self):
        """
        函数功能: 从 self.model_path 加载分类模型；不可用时由调用方使用检索兜底。
        :return:
        """
        if not os.path.exists(os.path.join(self.model_path, "config.json")):
            logger.warning("分类模型未安装，默认检索知识库；不会使用未经训练的分类头")
            return
        try:
            tokenizer_path = self.model_path if os.path.exists(
                os.path.join(self.model_path, 'vocab.txt')) else self.bert_path
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
        except (OSError, RuntimeError, ValueError):
            logger.exception("分类模型加载失败，使用检索兜底")
            self.model = None

    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"保存模型至: {self.model_path}")

    def preprocess_data(self, texts, labels):
        """
        函数功能: 对文本分词、截断和填充，并将标签转换为数字。
        :param texts: 待处理的文本列表.
        :param labels: 文本对应的标签列表.
        :return: 处理后的编码 (input_ids, attention_mask) 和 数字标签列表.
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def train_model(self, data_file=None):
        """
        函数功能: 使用标注数据训练 BERT 问题二分类模型。
        :param data_file: 数据集文件路径.
        :return: 无
        """
        data_file = data_file or os.path.join(rag_qa_path, 'classify_data', 'subject_scope_5000.json')
        if self.model is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, local_files_only=True)
            self.model = BertForSequenceClassification.from_pretrained(self.bert_path, num_labels=2,
                                                                       local_files_only=True)
            self.model.to(self.device)
        if (self.device.type == 'cuda' and torch.version.hip
                and torch.__version__.startswith('2.12.0')
                and torch.version.hip.startswith('7.14.')):
            # 此 ROCm 版本在 RX 7700 XT 上训练时，BERT 的 LayerNorm 梯度曾出现非有限值。
            self.model.set_attn_implementation('eager')
            for name, layer in list(self.model.named_modules()):
                if isinstance(layer, torch.nn.LayerNorm):
                    replacement = _ROCmTrainingLayerNorm(
                        layer.normalized_shape, eps=layer.eps,
                        elementwise_affine=layer.elementwise_affine,
                        device=layer.weight.device, dtype=layer.weight.dtype,
                    )
                    replacement.load_state_dict(layer.state_dict())
                    self.model.set_submodule(name, replacement)
            logger.info('ROCm 训练的 LayerNorm 使用 CPU，模型其余部分保持 GPU 运行')
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        with open(data_file, "r", encoding="utf-8") as f:
            raw = f.read()
            try:
                data = json.loads(raw)
                if isinstance(data, dict): data = [data]
            except json.JSONDecodeError:
                data = [json.loads(value) for value in raw.splitlines() if value.strip()]
        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)
        training_args = TrainingArguments(
            output_dir="./bert_results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=20,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            fp16=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[_CheckFiniteGradients()],
        )
        logger.info("开始训练 BERT 模型...")
        trainer.train()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise RuntimeError('训练后模型参数出现无效数值，未保存模型')
        self.save_model()
        self.evaluate_model(val_texts, val_labels)

    def compute_metrics(self, eval_pred):
        """
        函数功能: 计算分类任务的评估指标(准确率)
        :param eval_pred:  包含模型输出logits 和 真实标签的元素.
        :return: 准确率字典.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        """
        函数功能: 在给定文本和标签上评估模型, 输出分类报告和混淆矩阵.
        :param texts:  待评估的文本列表
        :param labels: 文本对应的真实标签(数字形式)
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels
        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            labels=[0, 1],
            target_names=["通用知识", "学科资料"],
            zero_division=0,
        ))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels, labels=[0, 1]))

    def predict_category(self, query):
        """
        函数功能: 对单个查询文本进行类别预测 -> 通用知识 或者 学科资料.
        :param query: 待分类的查询文本
        :return: 类别名称(文本标签)
        """
        if self.model is None:
            logger.debug("分类模型不可用，默认检索")
            return '学科资料'

        encoding = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(**encoding)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return '学科资料' if prediction == 1 else '通用知识'


if __name__ == '__main__':
    classifier = QueryClassifier(
        model_path=os.path.join(rag_qa_path, "models", "bert_query_classifier")
    )
    # classifier.model_path = os.path.join(rag_qa_path, 'models', 'bert_query_classifier')
    # data_file = os.path.join(rag_qa_path, 'classify_data', 'subject_scope_5000.json')
    # classifier.train_model(data_file)

    if classifier.model is None:
        print("模型未加载，请检查模型路径")
    else:
        test_queries = [
            "什么是人工智能？",
            "MySQL索引是干什么的？",
            "Python如何实现冒泡排序？",
            "今天为什么会下雨？",
            "数据库事务的隔离级别有哪些？",
            "什么是牛顿第二定律？",
            "Transformer的注意力机制是什么？",
            "如何提高睡眠质量？"
        ]

        for query in test_queries:
            result = classifier.predict_category(query)
            print(f"问题：{query}")
            print(f"预测：{result}")
            print("-" * 40)
