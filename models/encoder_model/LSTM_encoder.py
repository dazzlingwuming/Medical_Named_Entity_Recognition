import torch
from torch import nn
from transformers import BertConfig
from untils import Param
from models.LM_model.word2vec import Word2vecLMModel
from models.encoder_model._base import EncoderModel
import numpy as np
import os


class BILSTMEncoderModel(EncoderModel):
    def __init__(self,params):
        super(BILSTMEncoderModel,self).__init__(params)
        self.Lstm_layer = nn.LSTM(input_size = params.lm_bert_output_hidden_states,
                                      hidden_size = params.hidden_encoder_size,
                                      num_layers = params.lstm_layers,
                                      dropout = 0.0 if params.lstm_layers == 1 else params.lstm_dropout,#一层就不dropout
                                      batch_first = params.lstm_batch_first,
                                      bidirectional = params.lstm_bidirectional)
        self.layer_norm = nn.LayerNorm(params.hidden_encoder_size*(2 if params.lstm_bidirectional else 1)) if params.lstm_normal_init else nn.Identity()
        #因为后面的attention需要的维度是已经确定了，所以这里需要加一个层来使得输出是hidden_encoder_size维
        self.fc = nn.Linear(params.hidden_encoder_size*(2 if params.lstm_bidirectional else 1),params.hidden_encoder_size , bias=False)
    def forward(self,input_ids,input_mask , **kwargs):
        '''
        获取输入对应的每一个token向量
        :param input_ids:
        :param input_mask:按道理不需要了，因为在embed层已经mask了
        :param kwargs:
        :return:
        '''
        max_len = input_ids.shape[1]
        #提取特征
        lengths = input_mask.sum(1).long().cpu()
        embed = nn.utils.rnn.pack_padded_sequence(input_ids,lengths,batch_first=True,enforce_sorted=False)#去除padding部分
        output, (h_n, c_n) = self.Lstm_layer(input = embed)#[N,T,E]-->[N,T,2H] if bidirectional,此外h_n [ N,num_layers * num_directions, H],c_n [ N,num_layers * num_directions, H]
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True,total_length=max_len)#还原padding部分
        #如果embed层没有进行mask
        output = output * input_mask.unsqueeze(-1)
        #防止过拟合
        output = self.layer_norm(output)
        #特征融合
        output = self.fc(output)
        return output

if __name__=='__main__':
    params = Param(
        config=BertConfig.from_pretrained(r"../../pre_train_model/bert/config.json"),
    )
    model_LM = Word2vecLMModel(params)
    input_ids = torch.tensor([[1,2,3],[4,5,6]])
    input_mask = torch.tensor([[1,1,1],[1,1,0]])
    output = model_LM(input_ids,input_mask)
    print(output.shape)
    model_encoder = BILSTMEncoderModel(params)
    output_encoder = model_encoder(output,input_mask)
    print(output_encoder)
