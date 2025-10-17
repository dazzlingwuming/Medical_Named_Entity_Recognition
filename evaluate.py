import logging

import torch
from tqdm import tqdm
from metrics import SequenceLabelingMetrics
from untils import RunningAverage, Param


def evaluate_model(model, dataloader, param:Param ,device):
    model.eval()
    model.to(device)
    total_loss = 0
    total_correct = 0
    total_samples = 0
    avg_loss = RunningAverage()
    label_map = {v:k for k , v in torch.load(param.label2id).items()}
    all_true_labels = []  # 格式：List[List[int]]，每个子列表是一个样本的真实标签
    all_pred_labels = []  # 格式：List[List[int]]，每个子列表是一个样本的预测标签
    #遍历验证集，获取损失和准确率
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            input_mask = batch[2].to(device)

            _ , loss ,pred_label = model(input_ids, input_mask, labels=labels)#last_output（经过argmax输出的预测值 , loss）
            avg_loss.update(loss.item())

            mask = input_mask.bool()
            #真实值和预测值都需要在mask位置上计算
            true_labels = labels[mask].cpu()
            pred_labels = pred_label[mask].cpu()
            #累积到总列表中
            all_true_labels.append(true_labels.cpu().numpy().tolist())
            all_pred_labels.append(pred_labels.cpu().numpy().tolist())
            #计算评估指标
        metrics = {}
        metrics["loss"] = avg_loss()
        '''
        def calculate_ner_metrics(true_labels: List[List[int]], pred_labels: List[List[int]],
                          label_map: Dict[int, str], entity_types: List[str] = None) -> Dict:
        '''
        metrics["ner_metrics"] = SequenceLabelingMetrics.calculate_ner_metrics(all_true_labels,all_pred_labels, label_map)
        logging.info(f'-{",".join(map(str, metrics["ner_metrics"]))}')

    return metrics
