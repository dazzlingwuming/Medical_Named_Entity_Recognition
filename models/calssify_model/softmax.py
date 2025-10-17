from sympy.printing.pytorch import torch
from torch import nn
from models.calssify_model._base import CalssifyModel
from untils import Param

#因为后续需要对特殊层的参数进行微调，所以这里不使用nn.Sequential
class OutputFc(nn.Module):
    def __init__(self, param , input_size, layer_out):
        super(OutputFc, self).__init__()
        self.dropout = nn.Dropout(p=param.lstm_dropout)
        self.liner = nn.Linear(input_size, layer_out)
        self.layer_norm = nn.LayerNorm(layer_out)
        self.relu = nn.ReLU(inplace=True)# LayerNorm和ReLU有点冗余，一般可以选择其中一个

    def forward(self, input):
        return self.layer_norm(self.liner(self.dropout(input)))

class SoftmaxModel(CalssifyModel):
    '''
    n层全连接+softmax分类模型
    '''
    def __init__(self, params:Param):
        super(SoftmaxModel, self).__init__(params)
        self.params = params
        self.fc_layer_num = params.classify_fc_layer

        if self.fc_layer_num < 1:
            self.fc_layer = nn.Identity()
        else:
            fc_layers = []
            input_size = self.params.hidden_encoder_size
            for layer in self.params.classify_fc_hidden_size[:-1]:
                fc_layers.append(OutputFc(self.params, input_size, layer))
                input_size = layer
            fc_layers.append(nn.Linear(input_size, self.params.label_num))
            self.fc_layer = nn.Sequential(*fc_layers)#输出维度是[N,T,C]
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, input_mask,labels=None):
        '''
        前向传播，输出最终的分类结果，置信度，损失值等
        :param input_ids:
        :param input_mask:
        :param labels:真实标签
        :return:  [N,T],最终对应的分类结果，[N,T],对应的置信度，[1,],损失值
        -1.输出的feature是[N,T,C]的形式
        -2.针对每一个样本都进行遍历，选择25个置信度最高的作为最终的输出
        NOTE:
            但是由于softmax的输出是独立输出，每一个token都是独立分类的：
             - 第i个的输出结果是B-PER
             - 第i+1个的输出结果是O，但是这个的结果只受第i+1个token的影响，和第i个token的结果没有关系，但是从语义上来说，这两个结果是有关系的，结果应该是I-PER或者E-PER
             所以采用CRF等结构会更好
             前置模型中的序列结构是有考虑前后token的影响的，一定程度上面加强了token之间的联系，但是没有直接约束输出结果
        '''
        # input_ids [N,T,H]
        # input_mask [N,T]
        input_mask_weighted = input_mask.unsqueeze(-1).to(input_ids.dtype)  # [N,T,1]
        sequence_output = self.fc_layer(input_ids)*input_mask_weighted#[N,T,C]*[N,T,1] = [N,T,C]
        sequence_output_loss = sequence_output.permute(0, 2, 1)  # [N,C,T] 交叉熵损失需要[N,C,T]的输入
        pred = torch.argmax(sequence_output, dim=-1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(
                sequence_output_loss,
                labels
            )
        return sequence_output , loss , pred
