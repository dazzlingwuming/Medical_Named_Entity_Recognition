import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import numpy as np
from transformers import BertConfig

from models.encoder_model import EncoderModel
from untils import Param


def clones(module, N):
    "Produce N identical layers.模块复制，输入moudule ， N是重复次数，返回ModuleList"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size.残差连接"
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = d_model * 4
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute 'Scaled Dot Product Attention'
        query, key, value : batch_size, n_head, seq_len, dim of space
        注意这里的qkv是四维tensor
    """

    d_k = query.size(-1)
    # scores: batch_size, n_head, seq_len, seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MHPooling(nn.Module):
    def __init__(self, d_model, h, dropout=0.1, cuda=False):
        "Take in model size and number of heads."
        super(MHPooling, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)#q,k,v,output
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # auto-regressive
        attn_shape = (1, 3000, 3000)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        if cuda:
            self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1).cuda()
        else:
            self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1)

    def forward(self, x):
        "Implements Figure 2"

        nbatches, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        #多头注意力机制的qkv线性变换，取self.linears中的前三个线性层作为qkv的线性变换
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]

        # 2) Apply attention on all the projected vectors in batch.
        '''
        x：注意力机制的输出，形状为 (nbatches, h, seq_len, d_k)。
        self.attn：注意力权重，形状为 (nbatches, h, seq_len, seq_len)。
        '''
        x, self.attn = attention(query, key, value, mask=self.mask[:, :, :seq_len, :seq_len],
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout, cuda=False):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        """
        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)

        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())

        # To speed up
        idx = [i for j in range(self.ksize - 1, 10000, 1) for i in range(j - (self.ksize - 1), j + 1, 1)]
        if cuda:
            self.select_index = torch.LongTensor(idx).cuda()
            self.zeros = torch.zeros((self.ksize - 1, input_dim)).cuda()
        else:
            self.select_index = torch.LongTensor(idx)
            self.zeros = torch.zeros((self.ksize - 1, input_dim))

    def forward(self, x):
        nbatches, l, input_dim = x.shape
        x = self.get_K(x)  # b x seq_len x ksize x d_model
        #这里是控制序列长度的，用于加快计算速度
        batch, l, ksize, d_model = x.shape
        #获取局部窗口后，送入RNN，然后再reshape回来
        h = self.rnn(x.view(-1, self.ksize, d_model))[0][:, -1, :]
        return h.view(batch, l, d_model)

    def get_K(self, x):
        '''
        这里是为了加速而设计的局部窗口获取函数，具体思路是先在序列前面补ksize-1个0向量，然后通过index_select函数快速获取局部窗口
        :param x:
        :return:
        '''
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize * l])
        # 重新reshape，获取到局部窗口
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key


class LocalRNNLayer(nn.Module):
    "Encoder is made up of attconv and feed forward (defined below)"

    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout, cuda=False):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, dropout, cuda)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.connection(x, self.local_rnn)
        return x


class Block(nn.Module):
    """
    One Block
    """

    def __init__(self, input_dim, output_dim, rnn_type, ksize, N, h, dropout, cuda=False):
        super(Block, self).__init__()
        self.layers = clones(
            LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout, cuda), N)
        self.connections = clones(SublayerConnection(output_dim, dropout), 2)
        self.pooling = MHPooling(input_dim, h, dropout, cuda)
        self.feed_forward = PositionwiseFeedForward(input_dim, dropout)

    def forward(self, x):
        n, l, d = x.shape
        #local rnn --> 利用局部RNN提取局部特征
        for i, layer in enumerate(self.layers):
            x = layer(x)
        #global pooling + feed forward --> 利用多头池化提取全局特征
        x = self.connections[0](x, self.pooling)
        # feed forward
        x = self.connections[1](x, self.feed_forward)
        return x


class RTransformer(nn.Module):
    """
    The overal model
    """

    def __init__(self, d_model, rnn_type, ksize, n_level, n, h, dropout, cuda=False):
        '''
        d_model：模型的特征维度（所有层的输入输出维度均保持为d_model，保证残差连接可行）。
        rnn_type：局部 RNN 的类型（支持GRU、LSTM、RNN）。
        ksize：局部 RNN 的窗口大小（每个位置仅关注前ksize-1个位置 + 当前位置的局部序列）。
        n_level：模型堆叠的Block数量（即模型的 “深度”，每个Block是一组局部 RNN + 全局注意力的组合）。
        n：每个Block中堆叠的LocalRNNLayer数量（强化局部特征提取）。
        h：多头池化（MHPooling）的头数（需满足d_model % h == 0，保证每个头的维度均匀拆分）。
        dropout：dropout 概率（用于防止过拟合）。
        cuda：是否使用 GPU 加速（影响掩码和零向量的设备）。
        '''
        super(RTransformer, self).__init__()
        N = n
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, dropout)

        layers = []
        for i in range(n_level):
            layers.append(
                Block(d_model, d_model, rnn_type, ksize, N=N, h=h, dropout=dropout, cuda=cuda))
        self.forward_net = nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: [B,H,T,E]  B表示batch_size，H表示head数量，T表示序列长度，E表示特征维度
        :return:
        '''
        x = self.forward_net(x)
        return x

class RTransformerEncoder(EncoderModel):
    """
    RTransformer Encoder Wrapper
    """
    def __init__(self , param):
        super(RTransformerEncoder,self).__init__(params = param)
        self.rtransformer_layer = RTransformer(
            d_model=self.params.encoder_rtransformer_d_model,
            rnn_type=self.params.encoder_rtransformer_rnn_type,
            ksize=self.params.encoder_rtransformer_ksize,
            n_level=self.params.encoder_rtransformer_n_level,
            n=self.params.encoder_rtransformer_num_LocalRNNLayer,
            h=self.params.encoder_rtransformer_num_head,
            dropout=self.params.encoder_rtransformer_dropout,
            cuda= True if torch.cuda.is_available() else False
        )

    def forward(self , input_feature, input_mask, **kwargs):
        '''
        获取输入对应的每一个token向量
        E = self.params.config.hidden_embedding_size
        H = self.params.config.hidden_encoder_size
        :param input_feature: [N,T,E]表示N个样本，每一个样本具有T个token，每一个token对应E维向量
        :param input_mask: 每一个token是否有效，1表示有效，0表示无效
        :param kwargs: 意外参数
        :return: [N,T,H]表示N个样本，每一个样本具有T个token，每一个token对应H维向量
        '''

        output = self.rtransformer_layer(input_feature)
        # output = torch.
        return output

if __name__ == '__main__':
    # 测试 RTransformerEncoder
    config = BertConfig.from_pretrained(r'D:\github\Medical_Named_Entity_Recognition\pre_train_model\bert\config.json')
    params = Param(config = config)
    encoder = RTransformerEncoder(params)
    encoder.eval().cuda()
    # encoder = RTransformer(
    #         d_model=params.config.hidden_encoder_size,
    #         rnn_type='LSTM',
    #         ksize=3,
    #         n_level=2,
    #         n=2,
    #         h=4,
    #         dropout=0.1,
    #         cuda=
    #     )
    print(encoder)


    # 生成模拟输入
    input_feature = torch.rand(2, 10, 64).cuda()  # 形状：(批次大小, 序列长度, 特征维度)
    input_mask = torch.randint(0, 2, (2, 10)).cuda()  # 随机掩码（0表示填充，1表示有效）

    # 前向传播
    output = encoder(input_feature,input_mask)
    print("RTransformer Encoder 输出形状：", output.shape)  # 应为 (2, 10, 128)
