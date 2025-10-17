import torch
from torch import nn
from transformers import BertConfig
from untils import Param
from models.LM_model.word2vec import Word2vecLMModel
from models.encoder_model._base import EncoderModel
import numpy as np
import os


class IdCnnEncoderModel(EncoderModel):
    '''
    使用卷积神经网络来提前取文本的特征NLP中的文本序列特征提取，主要利用了N—gram的思想提取局部特征，+通过多层卷积和膨胀卷积来提取更大范围的特征
    NOTE:将文本序列长度当成图像的宽度，embedding维度当成图像的通道数，使用一维卷积来提取文本特征
    '''
    def __init__(self,params):
        super(IdCnnEncoderModel,self).__init__(params)
        layes = []
        in_filter = self.params.encoder_idcnn_in_filter
        for conv_param in self.params.encoder_idcnn_params:
            out_filter = conv_param.get("filters",in_filter)
            kernel_size = conv_param.get("kernel_size",3)
            dilation = conv_param.get("dilation",1)
            padding = conv_param.get("padding","same")
            '''
            - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})` N对应batch_size, C_in对应输入的embedding维度，L_in对应句子长度
            - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})` C_out对应输出的维度，L_out对应输出的句子长度
            '''
            layes.extend([
                nn.Conv1d(in_channels=in_filter,
                             out_channels=out_filter,
                             kernel_size=kernel_size,
                             dilation=dilation,
                             padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(out_filter)]
            )
            in_filter = out_filter
        #这里需要注意的是块的连接，要保证两个块的输入输出维度是一样的，所以这里in_filter没有变
        block = nn.Sequential(*layes)
        #block块重复
        layers = []
        for i in range(self.params.encoder_idcnn_num_blocks):
            layers.extend(
                [
                    block,
                    nn.ReLU(),
                    nn.BatchNorm1d(out_filter),
                ]
            )
        self.fc1 = nn.Linear(self.params.config.hidden_size, self.params.encoder_idcnn_in_filter)
        #全连接更改输入维度，匹配idcnn的输入维度

        self.idcnn_layers = nn.Sequential(*layers)
        #最后的全连接层
        self.fc2 = nn.Linear(in_filter , self.params.encoder_idcnn_output_size)

    def forward(self,input_ids,input_mask , **kwargs):
        '''
        获取输入对应的每一个token向量
        :param input_ids:[N,T,E]
        :param input_mask:按道理不需要了，因为在embed层已经mask了[N,T]
        :param kwargs:
        :return:
        '''
        input_features = self.fc1(input_ids)#[N,T,E]-->[N,T,C] C对应in_filter
        input_features = torch.permute(input_features, (0, 2, 1))#[N,E,T]-->[N,T,E]对应Conv1d的输入格式(N, C_{in}, L_{in})
        output_features = self.idcnn_layers(input_features)#[N,C,L] -->[N,C,L]
        output_features = torch.permute(output_features, (0, 2, 1))#[N,C,L]-->[N,L,C]对应[N,T,E]
        output_features = self.fc2(output_features)
        output_features = output_features * input_mask.unsqueeze(-1)
        return output_features


if __name__=='__main__':
    params = Param(
        config=BertConfig.from_pretrained(r"../../pre_train_model/bert/config.json"),
    )
    model_LM = Word2vecLMModel(params)
    input_ids = torch.tensor([[1,2,3],[4,5,6]])
    input_mask = torch.tensor([[1,1,1],[1,1,0]])
    output = model_LM(input_ids,input_mask)
    print(output.shape)
    model_encoder = IdCnnEncoderModel(params)
    output_encoder = model_encoder(output,input_mask)
    print(output_encoder)
