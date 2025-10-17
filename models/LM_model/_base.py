import torch
import torch.nn as nn
from untils import Param


class LMmodel(nn.Module):
    def __init__(self , params:Param):
        super(LMmodel, self).__init__()
        self.params = params
        self.freeze_params = params.freeze_lm_params


    def forward(self,input_ids,input_mask,**kwargs):
        '''
        获取输入对应的每一个token向量
        :param input_ids: token对应的ids，【N，T】表示N个样本，每一个样本具有T个token
        :param input_mask: 【N,T】表示N个样本，每一个样本具有T个token，1表示有效，0表示无效
        :param kwargs:其他相关参数
        :return:[N,T,E]表示N个样本，每一个样本具有T个token，每一个token对应E维向量,E等于self。hidden_size
        '''
        raise NotImplementedError("LM_model类需要重写forward方法")


    def freeze_model(self):
        '''
        冻结模型参数
        :return:
        '''
        print(f"当前实现冻结参数“{self.__class__.__name__}”")