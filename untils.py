import json
import os
from optparse import Option
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn
from transformers import BertConfig, NezhaConfig
from optimization import Adafactor

TAGS = ['I-手术', 'B-实验室检验', 'E-影像检查', 'E-解剖部位',
        'E-疾病和诊断', 'I-解剖部位', 'I-实验室检验', 'B-解剖部位',
        'I-疾病和诊断', 'I-药物', 'E-手术', 'B-影像检查', 'E-实验室检验',
        'I-影像检查', 'B-疾病和诊断', 'E-药物', 'B-手术', 'B-药物','O' , "<START>" , "<END>"]
class Param(object):
    '''
    参数类
    '''
    def __init__(self ,pre_model_type = "BERT" , ex_index = 1,params: dict= {}, config :Optional[Union[BertConfig,NezhaConfig]] = None):
        super(Param, self).__init__()
        self.pre_model_type = pre_model_type
        self.ex_index = ex_index
        #根目录
        self.root_path = Path(params.get("root_path" , os.path.abspath(os.path.dirname(__file__))))
        #数据集路径
        self.data_x_path = Path(params.get("data_x_path" , self.root_path / "data/json_data/x.txt"))
        self.data_y_path = Path(params.get("data_y_path" , self.root_path / "data/json_data/y.txt"))
        self.data_test_x_file = Path(params.get("data_test_x_file" , self.root_path / "data/json_data/test_x.txt"))
        self.data_test_y_file = Path(params.get("data_test_y_file" , self.root_path / "data/json_data/test_y.txt"))
        self.data_val_x_file = Path(params.get("data_val_x_file" , self.root_path / "data/json_data/val_x.txt"))
        self.data_val_y_file = Path(params.get("data_val_y_file" , self.root_path / "data/json_data/val_y.txt"))
        #参数路径
        self.label2id = Path(params.get("label2id" , self.root_path / "data/json_data/label_vocab.pkl"))
        self.label2id_vocab = { tag:i for i , tag in enumerate(TAGS)}
        self.vocab_path = Path(params.get("vocab_file" , self.root_path / "data/json_data/unique_chars.txt"))
        self.params_path = Path(params.get("params_path" , self.root_path / f"runs/experiment/{ex_index}"))
        self.params_path.mkdir(parents=True, exist_ok=True)
        #模型保存路径
        self.model_dir = Path(params.get("model_dir" , self.root_path / f"runs/model/{ex_index}"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        #bert模型保存路径
        self.bert_model_dir = Path(params.get("bert_model_dir" , self.root_path / f"pre_train_model/bert"))
        #albert模型保存路径
        #数据相关参数
        self.train_batch_size = params.get("train_batch_size" , 64)
        self.test_batch_size = params.get("test_batch_size" , 1)
        self.val_batch_size = params.get("val_batch_size" , 1)
        #pad
        self.pad = params.get("pad" , True)
        self.pad_id = params.get("pad_id" , "[PAD]")
        self.data_dir = Path(params.get("data_dir" , self.root_path / "data/json_data"))
        #序列长度
        self.max_len = config.max_position_embeddings
        #定义模型相关参数
        self.LM_model_name = params.get("LM_model_name" , "Word2vecLMModel")
        self.encoder_name = params.get("encoder_name" , "BILSTMEncoderModel")
        self.classify_name = params.get("classify_name" , "SoftmaxModel")
        self.lm_fusion_layers = params.get("lm_fusion_layers" , 4)#LM模型的融合层数
        self.lm_bert_output_hidden_states = params.get("lm_bert_output_hidden_states" , config.hidden_size)#是否输出bert模型的所有隐藏层
        self.config = config
        self.lstm_dropout = params.get("lstm_dropout", 0.1)  # 应该是数值，不是布尔值
        self.lstm_batch_first = params.get("lstm_batch_first", True)
        self.lstm_bidirectional = params.get("lstm_bidirectional", True)
        self.lstm_normal_init = params.get("lstm_normal_init", True)
        self.lstm_layers = params.get("lstm_layers", 1)
        #idcnn相关参数，注意多个卷积层必须后面的输入维度等于前面输出维度,或者在模型中进行调整
        self.encoder_idcnn_params = params.get("lstm_idcnn_params",[
            {
                "dilation": 1,
                "padding": "same",
                "filters": 128,
                "kernel_size": 3,
            }
            ,{
                "dilation": 1,
                "padding": "same",
                "filters": 256,
                "kernel_size": 4,
            },
            {
                "dilation": 1,
                "padding": "same",
                "filters": 128,
                "kernel_size": 5,
            }
        ])
        self.encoder_idcnn_in_filter = params.get("lstm_idcnn_in_filter", 128)
        self.encoder_idcnn_output_size = params.get("lstm_idcnn_output_size", 128)
        self.encoder_idcnn_kernel_size = params.get("lstm_idcnn_kernel_size", 3)
        self.encoder_idcnn_num_blocks = params.get("lstm_idcnn_num_blocks", 4)#重复的block块数
        self.encoder_rtransformer_rnn_type = params.get("encoder_rtransformer_rnn_type", 'LSTM')
        self.encoder_rtransformer_n_level = params.get("encoder_rtransformer_num_layers", 2)
        self.encoder_rtransformer_dropout = params.get("encoder_rtransformer_dropout", 0.1)
        self.encoder_rtransformer_d_model = params.get("encoder_rtransformer_d_model", 64)
        self.encoder_rtransformer_ksize = params.get("encoder_rtransformer_ksize", 3)
        self.encoder_rtransformer_num_LocalRNNLayer = params.get("encoder_rtransformer_num_LocalRNNLayer", 2)
        self.encoder_rtransformer_num_head = params.get("encoder_rtransformer_num_head", 4)
        self.hidden_encoder_size = params.get("hidden_encoder_size", 128)
        self.hidden_embedding_size = params.get("hidden_embedding_size", 64)#如果是word2vec模型，则需要定义词向量维度
        self.classify_fc_layer = params.get("classify_softmax_layer" , 2)
        self.classify_fc_hidden_size = params.get("classify_fc_hidden_size" , 128)#可以是多个全连接层的隐藏层维度，list等
        self.classify_fc_dropout = params.get("classify_dropout" , 0.1)
        self.label_num = params.get("label_num" , 19)
        if isinstance(self.classify_fc_hidden_size , int):
            self.classify_fc_hidden_size = [self.classify_fc_hidden_size]
        if self.classify_fc_hidden_size[-1] != self.label_num:
            self.classify_fc_hidden_size.append(self.label_num)
            print("自动添加分类类别数到最后一层")
        #梯度更新相关参数
        self.gradient_accumulation_steps = params.get("gradient_accumulation_steps" , 1)
        #训练相关参数
        self.log_steps = params.get("log_steps" , 10)#多少步打印一次日志
        self.num_epochs = params.get("num_epochs" , 5)#训练轮数
        self.freeze_lm_params = params.get("freeze_lm_params" , True)#是否冻结LM模型参数
        #优化器相关参数
        self.learning_rate = params.get("learning_rate" , 1e-3)#默认学习率
        self.lm_learning_rate = params.get("lm_learning_rate" , 5e-5)#embed层模型学习率
        self.encoder_learning_rate = params.get("encoder_learning_rate" , 5e-4)#encoder层模型学习率
        self.classify_learning_rate = params.get("classify_learning_rate" , 1e-4)#分类层模型学习率
        # self.lm_weight_decay = params.get("lm_weight_decay" , 1e-5)#embed层模型权重衰减
        # self.encoder_weight_decay = params.get("encoder_weight_decay" , 1e-5)#encoder层模型权重衰减
        # self.classify_weight_decay = params.get("classify_weight_decay" , 1e-5)#分类层模型权重衰减
        self.weight_decay = params.get("weight_decay" , 1e-3)#默认权重衰减
        #学习率调整
        self.lr_step_size = params.get("lr_step_size" , 3)
        self.lr_gamma = params.get("lr_gamma" , 0.1)
        #设备

    def load(self, json_path):
        '''
        从json文件中加载参数
        :param json_path:
        :return:
        '''
        assert os.path.exists(json_path), f"{json_path}文件不存在"
        with open(json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
            config = BertConfig(**params.get("config", {}))
            return Param(pre_model_type=params["pre_model_type"], ex_index=params["ex_index"], params=params,config = config)

    def save(self, json_path=None):
        '''
        保存参数到json文件中
        :param json_path:
        :return:
        '''
        if json_path is None:
            json_path = self.params_path / "params.json"
        params = {}
        with open(json_path, "w", encoding="utf-8") as f:
            for key, value in self.__dict__.items():
                if isinstance(value, Path):
                    value = str(value)
                if key == "config":
                    value = value.__dict__
                params[key] = value
            json.dump(params, f, ensure_ascii=False)

class RunningAverage:
    """计算最近N个值的平均值"""

    def __init__(self):
        self.step = 0
        self.total = 0

    def update(self, value):
        self.total += value
        self.step += 1

    def __call__(self):
        if self.step <= 0:
            return 0.0
        else:
            return self.total / float(self.step)

def save_model(state, is_best , checkpoint):#model和is_best是一个包含epoch,model_state_dict,optimizer_state_dict的字典
    filepath = os.path.join(checkpoint, "last.pth")
    if not os.path.exists(checkpoint):
        print(f"模型目录 {checkpoint} 不存在，创建中...")
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint, "best.pth")
        torch.save(state, best_filepath)
        print(f"已保存最佳模型到 {best_filepath}")

# 构建优化器
def bulid_optimizer(model:nn.Module, params):
    '''
    构建优化器
    这里主要是为了区分哪些参数需要权重衰减，哪些不需要
    例如，偏置项和LayerNorm层的参数通常不进行权重衰减
    这样做有助于防止过拟合，并提高模型的泛化能力
    :param model:
    :param params:
    :return:
    '''
    param_optimizer = list((n,p ) for n ,p in model.named_parameters() if p.requires_grad)
    #不同层使用不同的学习率
    lm_params = list((n , p) for n, p in param_optimizer if "LM_layer" in n)
    encoder_params = list((n , p) for n, p in param_optimizer if "encoder_layer" in n)
    classify_params = list((n , p) for n, p in param_optimizer if "classify_layer" in n)
    no_decay = ['bias', 'layer_norm', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        #lm层
        {'params': [p for n, p in lm_params if not any(nd in n for nd in no_decay)],
         "lr": params.lm_learning_rate,
         'weight_decay': params.weight_decay},
        {'params': [p for n, p in lm_params if any(nd in n for nd in no_decay)],
         "lr": params.lm_learning_rate,
         'weight_decay': 0.0},
        #encoder层
        {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
         "lr": params.encoder_learning_rate,
         'weight_decay': params.weight_decay},
        {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
         "lr": params.encoder_learning_rate,
         'weight_decay': 0.0},
        #分类层
        {'params': [p for n, p in classify_params if not any(nd in n for nd in no_decay)],
         "lr": params.classify_learning_rate,
         'weight_decay': params.weight_decay},
        {'params': [p for n, p in classify_params if any(nd in n for nd in no_decay)],
         "lr": params.classify_learning_rate,
         'weight_decay': 0.0},
    ]
    # optimizer = torch.optim.AdamW(**optimizer_grouped_parameters[0])#创建优化器时必须至少提供一个参数组；torch.optim.AdamW() 不能用空参数组初始化。
    # for i in range(1, len(optimizer_grouped_parameters)):
    #     optimizer.add_param_group(optimizer_grouped_parameters[i])
    #对于学习率预热和衰减，可以使用学习率调度器
    #这里直接调用huggingface的ADAMW优化器
    optimizer = Adafactor(
        params = optimizer_grouped_parameters,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,)
    return optimizer

if __name__ == "__main__":
    config = BertConfig.from_pretrained(r"pre_train_model/bert/config.json")
    params = {}
    params = Param(pre_model_type="BERT", ex_index=1, params=params, config=config)
    params.save()
    loaded_params = params.load(params.params_path / "params.json")
    print(loaded_params.__dict__)