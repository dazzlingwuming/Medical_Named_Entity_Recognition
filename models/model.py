from torch import nn
from transformers import BertConfig
from untils import Param
from .calssify_model import *
from .encoder_model import *
from .LM_model import *
import torch
import os

class NerModel(nn.Module):
    def __init__(self, params:Param):
        super(NerModel, self).__init__()
        self.params = params
        #创建三个部分，语言模型部分，编码层，分类模型部分
        self.LM_layer = eval(params.LM_model_name)(params)
        self.encoder_layer = eval(params.encoder_name)(params)
        self.classify_layer = eval(params.classify_name)(params)
        self.path = os.path.dirname(os.path.abspath(__file__))


    def forward(self, input_ids, input_mask ,labels=None):
        '''
        前向传播，输出最终的分类结果，置信度，损失值等
        :param input_ids:
        :param input_mask:
        :param labels:
        :return:
        '''
        #语言模型部分
        LM_output = self.LM_layer(input_ids, input_mask)
        #编码层部分
        encoder_output = self.encoder_layer(LM_output, input_mask)
        #分类模型部分
        classify_output = self.classify_layer(encoder_output, input_mask, labels)
        return classify_output

if __name__ == "__main__":
    params = {}
    cofig = BertConfig.from_pretrained(r"../pre_train_model/bert/config.json")
    params = Param(params=params,config= cofig)
    model = NerModel(params)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    input_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = model(input_ids, input_mask,labels=labels)
    print(output)
    print(model)