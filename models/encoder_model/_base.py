from torch import nn


class EncoderModel(nn.Module):
    def __init__(self,params):
        super(EncoderModel, self).__init__()
        self.params = params

    def forward(self,input_feature,input_mask,**kwargs):
        '''
        获取输入对应的每一个token向量
        E = self.params.config.hidden_embedding_size
        H = self.params.config.hidden_encoder_size
        :param input_feature: [N,T,E]表示N个样本，每一个样本具有T个token，每一个token对应E维向量
        :param input_mask: 每一个token是否有效，1表示有效，0表示无效
        :param kwargs: 意外参数
        :return: [N,T,H]表示N个样本，每一个样本具有T个token，每一个token对应H维向量
        '''
        raise NotImplementedError("EncoderModel类子类需要重写forward方法")
        pass