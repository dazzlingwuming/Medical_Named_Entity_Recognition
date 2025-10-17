'''
训练脚本
'''
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertConfig

from data.creat_data import DataLoaderMedical
from evaluate import evaluate_model
from untils import Param, RunningAverage, save_model, bulid_optimizer


def train_model(model:nn.Module, train_dataloader,val_dataloader,optimizer, params:Param):
    '''
    训练模型
    :param model: 模型
    :param train_dataloader: 训练数据加载器
    :param val_dataloader: 验证数据加载器
    :param optimizer: 优化器
    :param params: 参数
    :return:
    '''
    #参数持久化
    params.save(params.params_path / "params.json")
    #定义优化器
    optimizer = optimizer
    # #定义损失函数
    # criterion = nn.CrossEntropyLoss()
    # #定义学习率调度器
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step_size, gamma=params.lr_gamma)
    #如果有GPU则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_f1= 0 #最佳验证集f1值
    model.to(device)
    #开始训练
    for epoch in range(params.num_epochs):
        print(f"第{epoch + 1}轮训练开始")
        model.train()
        bar = tqdm(enumerate(train_dataloader))
        loss_avg = RunningAverage()
        for step, batch in bar:#batch是 input_ids, label_ids, input_mask, example_ids,split_to_original_id,但是只用到了input_ids, label_ids, input_mask
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            input_mask = batch[2].to(device)
            last_output , loss,_  = model(input_ids, input_mask, labels)
            #梯度累计需要损失平均
            if params.gradient_accumulation_steps > 1:
                loss = loss / params.gradient_accumulation_steps#损失缩放是为了梯度缩放，防止梯度更新过大
            loss.backward()
            #梯度更新有时候需要更多批次的累积，以产生更稳定的梯度
            if (step + 1) % params.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % params.log_steps == 0:
                loss_avg.update(loss)
                bar.set_postfix(ordered_dict={
                    "batch_loss": f"{loss.item():05.4f}",
                    "avg_loss": f"{loss_avg():05.4f}"
                                              })
        print(f"Epoch [{epoch + 1}/{params.num_epochs}] Training Loss: {loss_avg():.4f}")
        #验证模型
        if val_dataloader is not None:
            val_metrics = evaluate_model(model, val_dataloader, params, device)
            val_f1 = val_metrics["ner_metrics"]["micro"]["f1"]#验证集f1值
            improve_f1 = val_f1 - best_val_f1#当前f1值和最佳f1值的差值
            if improve_f1 > 0:
                best_val_f1 = val_f1
                #保存模型
                save_model(state={
                    'epoch': epoch + 1,
                    'model_state_dict': model,
                    'optimizer_state_dict': optimizer,
                    'best_val_f1': best_val_f1
                }, is_best=True, checkpoint=params.model_dir
                )
                # torch.save(model.state_dict(), params.model_dir / "best_model.pth")
                print(f"New best model saved with F1: {best_val_f1:.4f} in epoch {epoch + 1}")
            print(f"Validation Loss: {val_metrics['loss']:.4f}, F1: {val_f1:.4f}")
        # #更新学习率
        # scheduler.step()

# def train_evaluate_model(model:nn.Module, params:Param):
#     '''
#     训练并评估模型
#     :param model:
#     :param params:
#     :return:
#     '''
#     dataloader = DataLoaderMedical(params)
#     train_dataloader = dataloader.get_dataloard(data_sign="train")
#     val_dataloader = dataloader.get_dataloard(data_sign="val")
#
#     #训练和评估模型

def run():
    config = BertConfig.from_pretrained(r"pre_train_model/bert/config.json")
    params = {
        "LM_model_name": "BertLMModel",
        "encoder_name": "BILSTMEncoderModel",
        "classify_name": "SoftmaxModel",
        "classify_fc_layer": 2,
        "classify_fc_hidden_size": [128, 256],
        "classify_fc_dropout": 0.1,
        "label_num": 19,
        #数据相关参数
        "train_data_path": "./data/train.txt",
        "val_data_path": "./data/val.txt",
        "test_data_path": "./data/test.txt",
        "word2id": "./data/word2id.pkl",
        "batch_size": 32,
        "max_seq_len": 128,
        #训练相关参数
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "lr_step_size": 3,
        "lr_gamma": 0.1,
        "num_epochs": 5,
        "gradient_accumulation_steps": 1,
        "log_steps": 10,
        #模型保存相关参数
        "model_dir": "./model_checkpoints",
    }
    params = Param(ex_index=2,params=params, config=config)
    dataloader = DataLoaderMedical(params)
    train_dataloader = dataloader.get_dataloard(data_sign="train")
    val_dataloader = dataloader.get_dataloard(data_sign="val")

    from models.model import NerModel

    model = NerModel(params)
    optimizer = bulid_optimizer(model, params)
    train_model(model, train_dataloader, val_dataloader, optimizer, params)

if __name__ == '__main__':
    run()