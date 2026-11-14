"""BERT 问题二分类：训练、评估和预测通用知识 / 企业资料标签。"""

import argparse
import json
import os, sys
from datetime import datetime

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
    def __init__(self, model_path='../models/bert_enterprise_classifier'):
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
        self.label_map = {"通用知识": 0, "企业资料": 1}
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

    @staticmethod
    def _load_records(data_file):
        with open(data_file, 'r', encoding='utf-8') as stream:
            raw = stream.read()
            try:
                data = json.loads(raw)
                if isinstance(data, dict): data = [data]
            except json.JSONDecodeError:
                data = [json.loads(value) for value in raw.splitlines() if value.strip()]
        if not isinstance(data, list) or not data:
            raise ValueError(f'数据集 {data_file} 为空或格式不正确')
        for item in data:
            if 'query' not in item or 'label' not in item:
                raise ValueError(f'样本缺少 query/label 字段：{item}')
            if item['label'] not in ('通用知识', '企业资料'):
                raise ValueError(f'未知标签：{item["label"]}')
        return data

    @staticmethod
    def _split_records(data):
        """优先使用数据自带的 train/test 划分；同一 family 不允许跨集合。"""
        if any('split' in item for item in data):
            if not all(item.get('split') in ('train', 'test') for item in data):
                raise ValueError('启用 split 时所有样本都必须标记 train 或 test')
            family_split = {}
            for item in data:
                family = item.get('family') or item['query']
                if family in family_split and family_split[family] != item['split']:
                    raise ValueError(f'同一改写族跨训练/测试集：{family}')
                family_split[family] = item['split']
            train = [item for item in data if item['split'] == 'train']
            test = [item for item in data if item['split'] == 'test']
        else:
            train, test = train_test_split(
                data, test_size=0.2, random_state=42,
                stratify=[item['label'] for item in data],
            )
        if not train or not test:
            raise ValueError('训练集或测试集为空')
        if len({item['label'] for item in test}) < 2:
            raise ValueError('测试集必须同时包含企业资料和通用知识')
        return train, test

    def train_model(self, data_file=None):
        """
        函数功能: 使用标注数据训练 BERT 问题二分类模型。
        :param data_file: 数据集文件路径，默认使用按族切分的 v2 数据。
        :return: 训练与独立测试集指标
        """
        data_file = data_file or os.path.join(rag_qa_path, 'classify_data', 'enterprise_scope_v2.jsonl')
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

        records = self._load_records(data_file)
        train_records, test_records = self._split_records(records)
        train_texts = [item['query'] for item in train_records]
        train_labels = [item['label'] for item in train_records]
        test_texts = [item['query'] for item in test_records]
        test_labels = [item['label'] for item in test_records]

        logger.info('训练样本 %s 条（%s 个改写族），测试样本 %s 条（%s 个改写族）',
                    len(train_records), len({item.get('family') or item['query'] for item in train_records}),
                    len(test_records), len({item.get('family') or item['query'] for item in test_records}))
        train_encodings, train_label_ids = self.preprocess_data(train_texts, train_labels)
        test_encodings, test_label_ids = self.preprocess_data(test_texts, test_labels)
        train_dataset = self.create_dataset(train_encodings, train_label_ids)
        test_dataset = self.create_dataset(test_encodings, test_label_ids)
        training_args = TrainingArguments(
            output_dir=os.path.join(rag_qa_path, 'models', 'bert_training_runs'),
            num_train_epochs=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            fp16=False,
            report_to=[],
            dataloader_num_workers=0,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[_CheckFiniteGradients()],
        )
        logger.info("开始训练 BERT 模型...")
        trainer.train()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise RuntimeError('训练后模型参数出现无效数值，未保存模型')
        self.save_model()
        metrics = self.evaluate_model(test_texts, test_labels)
        metrics.update({
            'data_file': data_file,
            'train_samples': len(train_records),
            'test_samples': len(test_records),
            'train_families': len({item.get('family') or item['query'] for item in train_records}),
            'test_families': len({item.get('family') or item['query'] for item in test_records}),
            'label_counts': {
                'train': {label: sum(1 for item in train_records if item['label'] == label)
                          for label in ('企业资料', '通用知识')},
                'test': {label: sum(1 for item in test_records if item['label'] == label)
                         for label in ('企业资料', '通用知识')},
            },
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'command': 'python -m rag_qa.core.query_classifier --train',
        })
        metrics_path = os.path.join(self.model_path, 'training_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)
        logger.info('训练指标已保存至 %s', metrics_path)
        return metrics

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

    def evaluate_model(self, texts, labels, save_path=None):
        """
        函数功能: 在独立测试集上评估模型，返回精确率、召回率和误判样例。
        :param texts: 待评估的文本列表
        :param labels: 文本对应的真实标签（文本或数字形式）
        :param save_path: 可选的指标保存路径
        """
        if not texts:
            raise ValueError('测试集为空，无法评估')
        numeric_labels = [self.label_map[label] if isinstance(label, str) else int(label) for label in labels]
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        encodings = {key: value.to(self.device) for key, value in encodings.items()}
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(**encodings).logits.detach().cpu()
        probabilities = torch.softmax(logits, dim=-1).numpy()
        pred_labels = probabilities.argmax(axis=-1)
        true_labels = np.array(numeric_labels)
        report = classification_report(
            true_labels, pred_labels, labels=[0, 1],
            target_names=["通用知识", "企业资料"], output_dict=True, zero_division=0)
        matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).tolist()
        false_negatives = [
            dict(query=query, true_label='企业资料', predicted_label='通用知识',
                 confidence=float(probabilities[index][pred_labels[index]]))
            for index, (query, true, pred) in enumerate(zip(texts, true_labels, pred_labels))
            if true == 1 and pred == 0
        ]
        false_positives = [
            dict(query=query, true_label='通用知识', predicted_label='企业资料',
                 confidence=float(probabilities[index][pred_labels[index]]))
            for index, (query, true, pred) in enumerate(zip(texts, true_labels, pred_labels))
            if true == 0 and pred == 1
        ]
        metrics = {
            'accuracy': float(report['accuracy']),
            'enterprise_precision': float(report['企业资料']['precision']),
            'enterprise_recall': float(report['企业资料']['recall']),
            'enterprise_f1': float(report['企业资料']['f1-score']),
            'general_precision': float(report['通用知识']['precision']),
            'general_recall': float(report['通用知识']['recall']),
            'confusion_matrix': matrix,
            'false_negatives_enterprise_as_general': false_negatives,
            'false_positives_general_as_enterprise': false_positives,
        }
        logger.info('独立测试集准确率 %.4f，企业资料精确率 %.4f，召回率 %.4f',
                    metrics['accuracy'], metrics['enterprise_precision'], metrics['enterprise_recall'])
        logger.info('分类报告: %s', classification_report(
            true_labels, pred_labels, labels=[0, 1],
            target_names=["通用知识", "企业资料"], zero_division=0))
        logger.info('混淆矩阵: %s', matrix)
        if false_negatives:
            logger.warning('企业资料被误判为通用知识：%s', false_negatives)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as stream:
                json.dump(metrics, stream, ensure_ascii=False, indent=2)
        return metrics

    def predict_with_confidence(self, query):
        """
        函数功能: 返回类别、置信度和企业资料概率；模型缺失时显式标记不可用。
        """
        if self.model is None or self.tokenizer is None:
            return dict(available=False, label=None, confidence=0.0, enterprise_probability=0.0)
        encoding = self.tokenizer(
            query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        self.model.eval()
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**encoding).logits, dim=-1)[0]
        prediction = int(probabilities.argmax().item())
        return dict(
            available=True,
            label='企业资料' if prediction == 1 else '通用知识',
            confidence=float(probabilities[prediction].item()),
            enterprise_probability=float(probabilities[1].item()),
        )

    def predict_category(self, query):
        """
        函数功能: 对单个查询文本进行类别预测 -> 通用知识 或者 企业资料.
        :param query: 待分类的查询文本
        :return: 类别名称(文本标签)
        """
        result = self.predict_with_confidence(query)
        if not result['available']:
            logger.debug("分类模型不可用，默认检索")
            return '企业资料'
        return result['label']


def _binary_metrics(true_labels, predicted_labels):
    tp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 0)
    tn = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def compare_routing(classifier, data_file=None, threshold=0.8):
    """在同一独立测试集上比较“仅规则”和“规则+BERT”的企业问题识别效果。"""
    from rag_qa.core.rag_pipeline import (_requires_local_evidence, _DEFINITION_PATTERN,
                                          _COMPANY_MARKERS, _ENTERPRISE_TOPICS)
    data_file = data_file or os.path.join(rag_qa_path, 'classify_data', 'enterprise_scope_v2.jsonl')
    records = QueryClassifier._load_records(data_file)
    _, test_records = QueryClassifier._split_records(records)
    true_labels = [1 if item['label'] == '企业资料' else 0 for item in test_records]
    rule_predictions, combined_predictions = [], []
    false_negatives, false_positives = [], []
    rule_false_negatives = []
    for item, truth in zip(test_records, true_labels):
        query = item['query']
        rule = bool(_requires_local_evidence(query))
        result = classifier.predict_with_confidence(query)
        definition_only = bool(_DEFINITION_PATTERN.search(query)) and not (
            _COMPANY_MARKERS.search(query) or _ENTERPRISE_TOPICS.search(query))
        bert_enterprise = bool(
            result.get('available') and result['label'] == '企业资料'
            and result['confidence'] >= threshold and not definition_only)
        combined = rule or bert_enterprise
        rule_predictions.append(1 if rule else 0)
        combined_predictions.append(1 if combined else 0)
        if truth == 1 and not rule:
            rule_false_negatives.append(dict(query=query, confidence=result.get('confidence', 0.0),
                                             label=result.get('label')))
        if truth == 1 and not combined:
            false_negatives.append(dict(query=query, rule=rule, bert=bert_enterprise,
                                        confidence=result.get('confidence', 0.0),
                                        label=result.get('label')))
        if truth == 0 and combined:
            false_positives.append(dict(query=query, rule=rule, bert=bert_enterprise,
                                        confidence=result.get('confidence', 0.0),
                                        label=result.get('label')))
    report = {
        'threshold': threshold,
        'test_samples': len(test_records),
        'test_families': sorted({item.get('family') or item['query'] for item in test_records}),
        'rule_only': _binary_metrics(true_labels, rule_predictions),
        'rule_plus_bert': _binary_metrics(true_labels, combined_predictions),
        'false_negatives_rule_only': rule_false_negatives,
        'false_negatives_rule_plus_bert': false_negatives,
        'false_positives_rule_plus_bert': false_positives,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    output_dir = os.path.join(rag_qa_path, 'models', 'bert_enterprise_classifier')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'routing_comparison.json'), 'w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description='BERT 企业问题分类训练、预测与分流对比')
    parser.add_argument('--train', action='store_true', help='训练并保存分类模型')
    parser.add_argument('--data', help='训练数据文件（默认 enterprise_scope_v2.jsonl）')
    parser.add_argument('--predict', nargs='*', help='预测指定问题')
    parser.add_argument('--compare-routing', action='store_true', help='对比仅规则与规则+BERT')
    parser.add_argument('--threshold', type=float, default=0.8, help='BERT 反向触发企业检索的置信度阈值')
    args = parser.parse_args()

    classifier = QueryClassifier(model_path=os.path.join(rag_qa_path, 'models', 'bert_enterprise_classifier'))
    if args.train:
        metrics = classifier.train_model(args.data)
        print(json.dumps({key: metrics[key] for key in (
            'accuracy', 'enterprise_precision', 'enterprise_recall', 'enterprise_f1',
            'train_samples', 'test_samples')}, ensure_ascii=False, indent=2))
        return
    if args.compare_routing:
        compare_routing(classifier, args.data, args.threshold)
        return

    if classifier.model is None:
        print('模型未加载，请先运行 --train 或检查模型路径')
        return
    test_queries = args.predict or [
        'VPN 是什么？',
        '我们公司的 VPN 怎么申请？',
        '云帆科技的餐卡怎么充值？',
        '什么是人工智能？',
        '报销进度在哪里查询？',
    ]
    for query in test_queries:
        result = classifier.predict_with_confidence(query)
        print(f"问题：{query}")
        print(f"预测：{result['label']}（置信度 {result['confidence']:.4f}，"
              f"企业概率 {result['enterprise_probability']:.4f}）")
        print('-' * 40)


if __name__ == '__main__':
    main()
