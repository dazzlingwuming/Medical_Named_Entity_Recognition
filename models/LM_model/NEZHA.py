from torch import nn
from untils import Param
from models.LM_model._base import LMmodel
from transformers import BertTokenizer, BertConfig, AlbertModel , BertModel
import torch


class NeZhaLMModel(LMmodel):
    def __init__(self,params:Param):
        super(NeZhaLMModel, self).__init__(params)
        self.bert = NeZhaModel(self.params.config)
        #加权参数
        fusion_layers = min(self.params.lm_fusion_layers , self.params.config.num_hidden_layers)
        self.dym_weight = nn.Parameter(torch.ones(fusion_layers),requires_grad=False)
        # nn.init.xavier_uniform_(self.dym_weight)
        # 替换 xavier_uniform_ 为 uniform_
        nn.init.uniform_(self.dym_weight, a=-0.1, b=0.1)  # 使用均匀分布初始化

    def forward(self,input_ids,input_mask,**kwargs):
        z = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            output_hidden_states=True   # 是否输出所有层的隐藏状态
        )
        #将最后一层的隐藏状态加权求和
        z = sum(w * h for w, h in zip(self.dym_weight, z.hidden_states[-self.params.lm_fusion_layers:]))
        z = z*input_mask.unsqueeze(-1).to(z.dtype)
        return  z

    def freeze_model(self):
        '''
        冻结模型参数
        :return:
        '''
        if not self.freeze_params:
            return
        print(f"冻结参数“{self.__class__.__name__}”")
        for param in self.bert.parameters():
            param.requires_grad = False

if __name__=="__main__":
    params = Param(
        config= BertConfig.from_pretrained(r"C:\Users\lihaodong\.cache\huggingface\hub\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea\config.json"),
                 )
    model = BertLMModel(params)
    input_ids = torch.tensor([[1,2,3],[4,5,6]])
    input_mask = torch.tensor([[1,1,1],[1,1,0]])
    output = model(input_ids,input_mask)
    print(output)
    print(output.shape)  # 应该输出 torch.Size([2, 3, 128])
