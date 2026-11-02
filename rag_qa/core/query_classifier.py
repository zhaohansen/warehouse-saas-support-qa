"""BERT 意图分类：区分通用知识与专业咨询，含训练、评估与预测。"""

import json
import os, sys
current_dir= os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

import torch
from base.logger import logger
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class QueryClassifier:
    def __init__(self, model_path='../models/bert_query_classifier'):
        # 相对路径以当前源码目录为基准，避免从不同工作目录启动时加载失败。
        self.model_path = (
            os.path.normpath(os.path.join(current_dir, model_path))
            if not os.path.isabs(model_path) else model_path
        )
        bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {self.device}')
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        self.load_model()

    def load_model(self):
        """
        函数功能: 加载模型, 优先从self.model_path加载, 不存在则初始化基于 bert-base-chinese的新模型
        :return:
        """
        if os.path.exists(self.model_path):
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"加载模型: {self.model_path}")
        else:
            self.model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=2)
            self.model.to(self.device)
            # 新建模型也切换到推理模式，避免预测时启用 Dropout。
            self.model.eval()
            logger.info("初始化新 BERT 模型")

    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"保存模型至: {self.model_path}")

    def preprocess_data(self, texts, labels):
        """
        函数功能: 对文本进行分词, 阶段, 填充, 将标签转换为: 数字.
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

    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
        """
        函数功能: 训练BERT分类模型, 区分查询分类为'通用知识' 和 '专业咨询'
        :param data_file: 数据集文件路径.
        :return: 无
        """
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]
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
            logging_dir="./bert_logs",
            logging_steps=10,
            evaluation_strategy="epoch",
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
            compute_metrics=self.compute_metrics
        )
        logger.info("开始训练 BERT 模型...")
        trainer.train()
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
            target_names=["通用知识", "专业咨询"]
        ))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

    def predict_category(self, query):
        """
        函数功能: 对单个查询文本进行类别预测 -> 通用知识 或者 专业咨询.
        :param query: 待分类的查询文本
        :return: 类别名称(文本标签)
        """
        if self.model is None:
            logger.error("模型未训练或加载!")
            return '通用知识'

        encoding = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        encoding = {k: v.to(self.device) for k, v in encoding
        .items()}

        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(**encoding)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return '专业咨询' if prediction == 1 else '通用知识'

if __name__ == '__main__':
    query_classify = QueryClassifier()
    result = query_classify.predict_category('大数据分析师需要考取哪些证书')
    print(result)
