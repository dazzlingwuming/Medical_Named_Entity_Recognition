import torch
import models
from IPython import embed
from transformers import BertConfig
from data.dataloard_utils import Param
from models.LM_model import *
from models.calssify_model import *
from models.encoder_model import *
import numpy as np
from models.model import NerModel
import sys
import os

def t1():
    params = Param(
        config=BertConfig(hidden_embedding_size = 128,vocab_size = 1000)
    )
    model = Word2vecLMModel(params)
    input_ids = torch.tensor([[1,2,3],[4,5,6]])
    input_mask = torch.tensor([[1,1,1],[1,1,0]])
    output = model(input_ids,input_mask)
    print(output)
    print(output.shape)  # 应该输出 torch.Size([2, 3, 128])
    return output

def t2():
    params = Param(
        config=BertConfig(hidden_embedding_size=128, vocab_size=1000)
    )
    params.save()
    model_embed = Word2vecLMModel(params)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    input_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    output = model_embed(input_ids, input_mask)
    print(output.shape)
    print(output)
    model_encoder = BILSTMEncoderModel(params)
    output_encoder = model_encoder(output, input_mask)
    print(output_encoder.shape)
    print(output_encoder)

def t3():
    params = Param(
        config=BertConfig.from_pretrained(r"../pre_train_model/bert/config.json"),
    )
    params.save()
    model_embed = Word2vecLMModel(params)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    input_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    output = model_embed(input_ids, input_mask)
    # print(output.shape)
    model_encoder = BILSTMEncoderModel(params)
    output_encoder = model_encoder(output, input_mask)
    # print(output_encoder.shape)
    # params = Param(
    #     params={"classify_fc_layer": 2,
    #      "classify_fc_hidden_size": [128, 256],
    #      "classify_dropout": 0.1,
    #      "label_num": 19}
    # )
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    model_classify = SoftmaxModel(params)
    output_classify = model_classify(output_encoder, input_mask,labels = None)
    # print(output_classify)
    print(model_embed)
    print(model_encoder)
    print(model_classify)

def t4():
    params = Param(
        config=BertConfig.from_pretrained(r"../pre_train_model/bert/config.json"),
    )
    model = NerModel(params)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    input_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = model(input_ids, input_mask,labels=labels)
    # print(output)
    print(model)
    param_optimizer = list(model.named_parameters())
    print(param_optimizer)

def t5():
    bert_path= r"D:\github\Medical_Named_Entity_Recognition\pre_train_model\bert\pytorch_model.bin"
    params = Param(
        config=BertConfig.from_pretrained(r"C:\Users\lihaodong\.cache\huggingface\hub\models--clue--albert_chinese_tiny\snapshots\654acaf73c361ad56e4f4b1e2bb0023cbb1872b2\config.json"),
        params={
            "LM_model_name": "AlbertLMModel",
            "bert_model_dir": bert_path,
        }
    )
    model = NerModel(params)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    input_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = model(input_ids, input_mask,labels=labels)
    # print(output)
    print(model)
    # param_optimizer = list(model.named_parameters())
    # print(param_optimizer)
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # print(optimizer_grouped_parameters)

def t6():
    bert_path = r"C:\Users\lihaodong\.cache\huggingface\hub\models--google-bert--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea\pytorch_model.bin"
    params = Param(
        config=BertConfig.from_pretrained(
            r"C:\Users\lihaodong\.cache\huggingface\hub\models--google-bert--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea\config.json"),
        params={
            "LM_model_name": "BertLMModel",
            "bert_model_dir": bert_path,
        }
    )
    model = NerModel(params)
    # print(model)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    input_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = model(input_ids, input_mask, labels=labels)
    print(output)
    print(model)
    # param_optimizer = list(model.named_parameters())
    # print(param_optimizer)
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # print(optimizer_grouped_parameters)

def t7():
    bert_path = r"C:\Users\lihaodong\.cache\huggingface\hub\models--google-bert--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea\pytorch_model.bin"
    params = Param(
        config=BertConfig.from_pretrained(
            r"C:\Users\lihaodong\.cache\huggingface\hub\models--google-bert--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea\config.json"),
        params={
            "LM_model_name": "BertLMModel",
            "bert_model_dir": bert_path,
            "encoder_name": "IdCnnEncoderModel",
        }
    )
    model = NerModel(params)
    print(model)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    input_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = model(input_ids, input_mask, labels=labels)
    print(output)
    print(model)
    # param_optimizer = list(model.named_parameters())
    # print(param_optimizer)
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # print(optimizer_grouped_parameters)\
def t8():
    bert_path = r"C:\Users\lihaodong\.cache\huggingface\hub\models--google-bert--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea\pytorch_model.bin"
    config = BertConfig.from_pretrained(
            r"C:\Users\lihaodong\.cache\huggingface\hub\models--google-bert--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea\config.json")
    params = Param(
        config=config,
        params={
            "LM_model_name": "BertLMModel",
            "bert_model_dir": bert_path,
            "encoder_name": "RTransformerEncoder",
            "encoder_rtransformer_d_model":config.hidden_size,
        }
    )
    model = NerModel(params).cuda()
    print(model)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
    input_mask = torch.tensor([[1, 1, 1], [1, 1, 0]]).cuda()
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
    output = model(input_ids, input_mask, labels=labels)
    print(output)



if __name__ == '__main__':
    t8()