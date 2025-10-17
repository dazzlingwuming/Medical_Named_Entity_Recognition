'''
分类模型
'''

import torch
import torch.nn as nn
import numpy as np

class CalssifyModel(nn.Module):
    def __init__(self, params):
        super(CalssifyModel, self).__init__()
        self.params = params

    def forward(self, input_ids, input_mask ,labels=None):
        '''
        前向传播，输出最终的分类结果，置信度，损失值等
        :param input_ids:
        :param input_mask:
        :param labels:真实标签
        :return:  [N,T],最终对应的分类结果，[N,T],对应的置信度，[1,],损失值
        '''
        raise NotImplementedError("子分类器没有实现前向传播")
