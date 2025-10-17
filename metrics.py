"""
深度学习常用评估指标工具集
包含分类、回归、序列标注等任务的评估指标
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import List, Dict, Tuple, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from untils import Param

from sympy.physics.vector.printing import params


class ClassificationMetrics:
    """分类任务评估指标"""

    @staticmethod
    def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        计算二分类任务的各项指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率（用于AUC计算）

        Returns:
            包含各项指标的字典
        """
        metrics = {}

        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # 混淆矩阵相关指标
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假正率

        # AUC（如果提供了概率）
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.0

        return metrics

    @staticmethod
    def calculate_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                     y_prob: Optional[np.ndarray] = None,
                                     average_methods: List[str] = ['macro', 'micro', 'weighted']) -> Dict:
        """
        计算多分类任务的各项指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            average_methods: 平均方法列表

        Returns:
            包含各项指标的字典
        """
        metrics = {}

        # 准确率
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # 不同平均方法的指标
        for avg in average_methods:
            metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)

        # 每个类别的详细指标
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['class_report'] = class_report

        # 多分类AUC（如果提供了概率）
        if y_prob is not None and len(np.unique(y_true)) > 2:
            try:
                # 使用one-vs-rest策略计算多分类AUC
                metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
            except:
                metrics['auc_ovr'] = 0.0
                metrics['auc_ovo'] = 0.0

        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: List[str] = None,
                              normalize: bool = True,
                              title: str = 'Confusion Matrix') -> plt.Figure:
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            normalize: 是否归一化
            title: 图表标题

        Returns:
            matplotlib图表对象
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)

        return fig


class RegressionMetrics:
    """回归任务评估指标"""

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        计算回归任务的各项指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            包含各项指标的字典
        """
        metrics = {}

        # 基础回归指标
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

        # 相对误差指标
        abs_error = np.abs(y_true - y_pred)
        metrics['mape'] = np.mean(abs_error / (np.abs(y_true) + 1e-8)) * 100  # 平均绝对百分比误差
        metrics['smape'] = 2.0 * np.mean(abs_error / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

        # 统计信息
        metrics['max_error'] = np.max(abs_error)
        metrics['std_error'] = np.std(y_true - y_pred)

        return metrics

    @staticmethod
    def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray,
                                title: str = 'Regression Results') -> plt.Figure:
        """
        绘制回归结果可视化

        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题

        Returns:
            matplotlib图表对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 散点图：预测值 vs 真实值
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('True vs Predicted')

        # 残差图
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')

        plt.tight_layout()
        return fig


class SequenceLabelingMetrics:
    """序列标注任务评估指标"""

    @staticmethod
    def extract_entities(labels: List[int], label_map: Dict[int, str]) -> List[Tuple]:
        """
        从标签序列中提取实体

        Args:
            labels: 标签序列
            label_map: 标签映射字典

        Returns:
            实体列表 [(start, end, entity_type), ...]
        """
        entities = []
        current_entity = None

        for i, label_id in enumerate(labels):
            label = label_map[label_id]

            if label.startswith('B-'):
                # 开始新实体
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = (i, i, label[2:])
            elif label.startswith('I-'):
                # 继续当前实体
                if current_entity is not None and current_entity[2] == label[2:]:
                    current_entity = (current_entity[0], i, current_entity[2])
                else:
                    # 不匹配的I标签，结束当前实体
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = None
            else:
                # O标签，结束当前实体
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = None

        # 添加最后一个实体
        if current_entity is not None:
            entities.append(current_entity)

        return entities

    @staticmethod
    def calculate_ner_metrics(true_labels: List[List[int]], pred_labels: List[List[int]],
                              label_map: Dict[int, str], entity_types: List[str] = None) -> Dict:
        """
        计算NER任务的精确率、召回率、F1分数

        Args:
            true_labels: 真实标签序列列表
            pred_labels: 预测标签序列列表
            label_map: 标签映射字典
            entity_types: 要评估的实体类型列表，None表示所有类型

        Returns:
            包含各项指标的字典
        """
        # 统计TP, FP, FN
        stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        for true_seq, pred_seq in zip(true_labels, pred_labels):
            true_entities = set(SequenceLabelingMetrics.extract_entities(true_seq, label_map))
            pred_entities = set(SequenceLabelingMetrics.extract_entities(pred_seq, label_map))

            # 按实体类型统计
            for entity_type in set([e[2] for e in true_entities] + [e[2] for e in pred_entities]):
                if entity_types is not None and entity_type not in entity_types:
                    continue

                true_entities_type = set(e for e in true_entities if e[2] == entity_type)
                pred_entities_type = set(e for e in pred_entities if e[2] == entity_type)

                # 计算TP, FP, FN
                tp = len(true_entities_type & pred_entities_type)
                fp = len(pred_entities_type - true_entities_type)
                fn = len(true_entities_type - pred_entities_type)

                stats[entity_type]['tp'] += tp
                stats[entity_type]['fp'] += fp
                stats[entity_type]['fn'] += fn

        # 计算每个实体类型的指标
        metrics = {}
        for entity_type, counts in stats.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn  # 真实实体数量
            }

        # 计算宏平均和微平均
        macro_precision = np.mean([m['precision'] for m in metrics.values()])
        macro_recall = np.mean([m['recall'] for m in metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in metrics.values()])

        total_tp = sum(stats[et]['tp'] for et in stats)
        total_fp = sum(stats[et]['fp'] for et in stats)
        total_fn = sum(stats[et]['fn'] for et in stats)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

        metrics['macro'] = {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }

        metrics['micro'] = {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        }

        return metrics


class ModelEvaluator:
    """模型评估器 - 综合评估工具"""

    def __init__(self, task_type: str = 'classification'):
        """
        初始化评估器

        Args:
            task_type: 任务类型 ['classification', 'regression', 'sequence_labeling']
        """
        self.task_type = task_type
        self.history = []

    def evaluate_classification(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                                device: torch.device, class_names: List[str] = None) -> Dict:
        """
        评估分类模型

        Args:
            model: 模型
            dataloader: 数据加载器
            device: 设备
            class_names: 类别名称

        Returns:
            评估结果字典
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(device) if 'input_ids' in batch else batch[0].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else batch[1].to(device)

                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 取第一个输出

                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(outputs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算指标
        if len(np.unique(all_labels)) == 2:
            metrics = ClassificationMetrics.calculate_binary_metrics(all_labels, all_preds, all_probs[:, 1])
        else:
            metrics = ClassificationMetrics.calculate_multiclass_metrics(all_labels, all_preds, all_probs)

        # 保存评估历史
        self.history.append({
            'epoch': len(self.history),
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels
        })

        return metrics

    def evaluate_regression(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                            device: torch.device) -> Dict:
        """
        评估回归模型

        Args:
            model: 模型
            dataloader: 数据加载器
            device: 设备

        Returns:
            评估结果字典
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(device) if 'input_ids' in batch else batch[0].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else batch[1].to(device)

                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 取第一个输出

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()

        metrics = RegressionMetrics.calculate_regression_metrics(all_labels, all_preds)

        # 保存评估历史
        self.history.append({
            'epoch': len(self.history),
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels
        })

        return metrics

    def plot_training_history(self, metrics_to_plot: List[str] = None) -> plt.Figure:
        """
        绘制训练历史

        Args:
            metrics_to_plot: 要绘制的指标列表

        Returns:
            matplotlib图表对象
        """
        if not self.history:
            raise ValueError("No evaluation history available")

        epochs = [h['epoch'] for h in self.history]

        if metrics_to_plot is None:
            # 自动选择要绘制的指标
            metrics_to_plot = list(self.history[0]['metrics'].keys())
            metrics_to_plot = [m for m in metrics_to_plot if not m.startswith('class_report')]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics_to_plot):
            values = [h['metrics'].get(metric, 0) for h in self.history]
            ax.plot(epochs, values, 'b-o', linewidth=2, markersize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} over Epochs')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# 使用示例
if __name__ == "__main__":
    # # 分类任务示例
    # print("=== 分类指标示例 ===")
    # y_true_cls = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    # y_pred_cls = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    # y_prob_cls = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.7, 0.8, 0.3, 0.4])
    #
    # binary_metrics = ClassificationMetrics.calculate_binary_metrics(y_true_cls, y_pred_cls, y_prob_cls)
    # print("二分类指标:", binary_metrics)
    #
    # # 回归任务示例
    # print("\n=== 回归指标示例 ===")
    # y_true_reg = np.array([1.2, 2.4, 3.1, 4.8, 5.0])
    # y_pred_reg = np.array([1.1, 2.1, 3.3, 4.5, 5.2])
    #
    # reg_metrics = RegressionMetrics.calculate_regression_metrics(y_true_reg, y_pred_reg)
    # print("回归指标:", reg_metrics)

    # 序列标注示例
    label_map = {v:k for k , v in torch.load(param.label2id).items()}
    true_labels = [[0, 1, 2, 0, 3, 4], [1, 2, 0, 0, 3, 4]]
    pred_labels = [[0, 1, 2, 0, 0, 0], [1, 2, 0, 3, 4, 0]]

    ner_metrics = SequenceLabelingMetrics.calculate_ner_metrics(true_labels, pred_labels, label_map)
    print("NER指标:", ner_metrics)