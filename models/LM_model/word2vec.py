from torch import nn
from untils import Param
from models.LM_model._base import LMmodel
from transformers import BertTokenizer, BertConfig
import torch


class Word2vecLMModel(LMmodel):
    def __init__(self,params:Param):
        super(Word2vecLMModel, self).__init__(params)
        self.hidden_size = params.hidden_embedding_size
        self.vocab_size = params.config.vocab_size
        #实现gensim训练的word2vec向量作为embedding层的初始权重
        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)

    def forward(self,input_ids,input_mask,**kwargs):
        z = self.word_embeddings(input_ids)
        z = z*input_mask.unsqueeze(-1)
        return  z
if __name__=="__main__":
    params = Param(
        config=BertConfig(hidden_embedding_size = 128,vocab_size = 1000)
    )
    model = Word2vecLMModel(params)
    input_ids = torch.tensor([[1,2,3],[4,5,6]])
    input_mask = torch.tensor([[1,1,1],[1,1,0]])
    output = model(input_ids,input_mask)
    print(output)
    print(output.shape)  # 应该输出 torch.Size([2, 3, 128])
