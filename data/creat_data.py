# -*- coding: utf-8 -*-
'''
对数据集进行处理
'''
import re
from collections import OrderedDict
import pickle as pkl
from distutils.command.config import dump_file
from linecache import cache
import numpy
import torch
from torch.utils.data import DataLoader , Dataset , RandomSampler,SequentialSampler
import torch.nn as nn
import os
import json
import random
import numpy as np
import pandas as pd
from fontTools.misc.psOperators import ps_string
from common import creat_file
from torchtext import vocab
from .dataloard_utils import read_txt, convert_examples_to_features, Param
from .dataloard_utils import read_json
from data.dataloard_utils import read_examples
from transformers import BertTokenizer


#加载txt文件转换成json格式
def txt_convert_json(file_path):
    if not file_path.endswith('.txt'):
        raise ValueError("文件格式错误，请提供txt文件")
    json_list = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                json_list.append(json.loads(line))

        # 写入为标准 JSON 数组
        with open("json_data/subtask1_training_part1.json", "w", encoding="utf-8") as f:
            json.dump(json_list, f, ensure_ascii=False, indent=2)

        print("转换完成，已保存为 subtask1_training_part1.json")

#划分数据并写入文件
def split_data_2_x_y(datas):#读取json数据返回的原始xy，是存在着多个例子
    x = []
    y = []
    for data in datas:#读取每一个例子，格式是原文本+实体标签
        x_hat = data["originalText"]
        x_hat_len = len(x_hat)
        x.append(x_hat)
        y_hat = create_y(data, x_hat_len)#看一下数据是否正确
        # create_y(data["entities"] , x_hat_len)
        y.append(y_hat)
    creat_file("json_data/对比.txt")
    with open("json_data/对比.txt", 'w', encoding='utf-8') as f:
        for token , label in zip(x[1], y[1]):
            f.writelines(f"{token} {label}\n")
    print(" ")
    return x, y
#将x和y写入文件
def write_file(path_x , path_y , x , y):
    creat_file(path_x)
    creat_file(path_y)
    with open(path_x, 'w', encoding='utf-8') as f:
        for i in range(len(x)):
            f.writelines(x[i] + '\n')
    with open(path_y, 'w', encoding='utf-8') as f:
        for i in range(len(y)):
            f.writelines(' '.join(y[i]) + '\n')
    print(f'数据已写入{path_x}和{path_y}文件中')
#根据实体标签创建y
def create_y(datas ,x_hat_len):#这里需要传入的是每一个例子，对应存在着多个实体标签
    #这里采用BIOES
    y = ["O"] * x_hat_len
    for data in datas["entities"]:
        start_pos = data["start_pos"]
        end_pos = data["end_pos"]
        label_type = data["label_type"]
        if start_pos == end_pos-1:
            y[start_pos] = "S-" + label_type
        else:
            y[start_pos] = "B-" + label_type
            y[start_pos +1 : end_pos-1] = ["I-" + label_type] * (end_pos - start_pos -2)
            y[end_pos-1] = "E-" + label_type
        # print(f'{datas["originalText"][start_pos:end_pos+1]}对应的是{y[start_pos:end_pos+1]}')
    return y
#创建标签词典
def create_labels_vocab(read_path , out_path):
    labels = set()
    label_vocab = {}
    creat_file(out_path)
    y = read_txt(read_path)
    for line in y:
        line = line.strip().split(" ")
        for label in line:
            labels.add(label)
    for id ,label in enumerate(labels):
        label_vocab[label] = id
    dump_file = os.path.join(out_path, "label_vocab.pkl")
    torch.save(label_vocab, dump_file)
#创建token和id的映射词典
def create_token_vocab(read_path , out_path , default_tokens ={'<unk>':0, '<pad>':1}  ):
    token_vocab = {}
    default_tokens_len = len(default_tokens)
    creat_file(out_path)
    with open(read_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        all_text = ''.join([line.strip() for line in lines])
        unique_chars = set(all_text)
        # with open("json_data/unique_chars.txt", 'w', encoding='utf-8') as fw:
        #     for char in unique_chars:
        #         fw.writelines(f"{char}\n")
        '''上面是构建一个包含所有字符的文件，用到后面的tokenizer'''
        for idx, char in enumerate(unique_chars):
            token_vocab[char] = idx+ default_tokens_len
    token_vocab = OrderedDict(token_vocab)
    token_vocab = vocab.vocab(token_vocab)
    for token , id in default_tokens.items():
        if token not in token_vocab:
            token_vocab.insert_token(token , id)
    token_vocab.set_default_index(token_vocab['<unk>'])
    dump_file = os.path.join(out_path, "token_vocab.pkl")
    torch.save(token_vocab, dump_file)
    return token_vocab

def split_text(text, max_len, split_pat=r'([，。]"?)|([.!?]"?)|(\s+)', greedy=False):
    """
    文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本

    Arguments:
        text {str} -- 原始文本
        max_len {int} -- 最大长度
        split_pat {str} -- 分割模式正则表达式
        greedy {bool} -- 是否使用贪婪模式

    Returns:
        list -- 分割后的文本片段列表
        list -- 每个片段的起始位置列表
    """
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text.strip())

    # 文本小于最大长度
    if len(text) <= max_len:
        return [text], [0]

    # 直接按字符分割，不依赖标点
    segments = []
    starts = []
    start = 0

    while start < len(text):
        # 计算当前片段的结束位置
        end = min(start + max_len, len(text))

        # 如果正好在结尾，直接添加
        if end == len(text):
            segments.append(text[start:])
            starts.append(start)
            break

        # 尝试在标点处分割
        segment = text[start:end]

        # 查找最后一个分割点
        last_split_pos = -1

        # 查找标点符号
        for i in range(len(segment) - 1, -1, -1):
            if re.match(split_pat, segment[i]):
                last_split_pos = i
                break

        # 如果没有找到标点，就在max_len处强制分割
        if last_split_pos == -1:
            last_split_pos = len(segment) - 1

        # 确保分割点有效
        actual_end = start + last_split_pos + 1

        # 添加片段
        segments.append(text[start:actual_end])
        starts.append(start)

        # 更新起始位置
        start = actual_end

    return segments, starts

class DataSetMedical(Dataset):
    def __init__(self, feature):#读取词典,这里的
        super(DataSetMedical, self).__init__()
        self.feature = feature

    def __getitem__(self, index):
        return self.feature[index]

    def __len__(self):
        return len(self.feature)

class DataLoaderMedical(object):
    def __init__(self, params):
        super(DataLoaderMedical, self).__init__()
        self.data_x_file = params.data_x_path
        self.data_y_file = params.data_y_path
        self.data_test_x_file = params.data_test_x_file
        self.data_test_y_file = params.data_test_y_file
        self.data_val_x_file = params.data_val_x_file
        self.data_val_y_file = params.data_val_y_file
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.val_batch_size = params.val_batch_size
        self.tokenizer = BertTokenizer(vocab_file=params.vocab_path, do_lower_case=True)
        self.label2id = torch.load(params.label2id)
        self.max_len = params.max_len
        self.pad = True
        self.pad_id = params.pad_id
        self.data_dir = params.data_dir

    @staticmethod
    def collate_fn(features):
        '''
        自定义的collate_fn函数
        将一个批次的features合并成一个batch
        :param batch:example_ids,input_ids ,label_ids ,input_mask , split_to_original_id
        :return:
        '''
        # batch是一个列表，列表的每一个元素是dataset的__getitem__方法的返回值
        # 假设dataset的__getitem__方法返回的是一个元组(x, y)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)#输入x
        label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)#输出y
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)#mask
        example_ids = torch.tensor([f.example_ids for f in features], dtype=torch.long)#样本id
        split_to_original_id = torch.tensor([f.split_to_original_id for f in features], dtype=torch.long)#分割后文本对应的原始文本id
        tensors = [ input_ids, label_ids, input_mask, example_ids,split_to_original_id]
        return tensors

    def get_feature(self,data_sign):
        '''
        获取特征
        :return:feature:[list] InputExamples-->InputFeatures
        '''
        cache_path = os.path.join(self.data_dir, f"{data_sign}.cache.{self.max_len}")
        if os.path.exists(cache_path):
            print(f"加载缓存的{data_sign}数据中")
            feature = torch.load(cache_path)
            return feature
        #1.加载InputExamples
        print(f"加载{data_sign}数据中")
        if data_sign in ["train","test","val"]:
            if data_sign == "train":
                examples = read_examples(self.data_x_file , self.data_y_file)
            elif data_sign == "test":
                examples = read_examples(self.data_test_x_file , self.data_test_y_file)
            elif data_sign == "val":
                examples = read_examples(self.data_val_x_file , self.data_val_y_file)
        else:
            raise ValueError("data_sign取值错误，仅支持train/test/val")
        #2.将InputExamples转换成InputFeatures
        print(f"转换{data_sign}数据中")
        feature = convert_examples_to_features(
            examples, word2id={},tag2id=self.label2id , tokenizer=self.tokenizer, max_seq_length=self.max_len,
            data_sign="train", pad_sign=self.pad, pad_token=self.pad_id,
        )
        if cache_path:
            torch.save(feature, cache_path)
        return feature

    def get_dataloard(self,data_sign = "train"):
        '''
        获取dataloard对象
        :param data_sign:
        :return:
        '''
        #获取特征
        feature = self.get_feature(data_sign)
        dataset = DataSetMedical(feature)
        #创建dataloader对象
        if data_sign == "train":
            batch_size = self.train_batch_size
            datasampler = RandomSampler(dataset)
        elif data_sign == "test":
            batch_size = self.test_batch_size
            datasampler = SequentialSampler(dataset)
        elif data_sign == ("val"):
            batch_size = self.val_batch_size
            datasampler = SequentialSampler(dataset)
        else:
            raise ValueError("data_sign取值错误，仅支持train/test/val")
        dataloader = DataLoader(dataset , batch_size=batch_size , sampler=datasampler , drop_last=False,collate_fn=self.collate_fn)
        return dataloader



if __name__ == '__main__':
    # txt文件转换成json格式
    # a = txt_convert_json("yidu-s4k/subtask1_training_part1.txt")
    # 读取json文件
    # data = read_json("json_data/subtask1_training_part1.json")
    # split_data_2_x_y(data )#划分数据集并写入文件
    # create_labels_vocab("json_data/y.txt" , "json_data")#保存标签词典
    # labels_vocab = torch.load("json_data/label_vocab.pkl")
    # create_token_vocab("json_data/x.txt" , "json_data")#保存token词典
    # token_vocab = torch.load("json_data/token_vocab.pkl")
    # print(token_vocab)
    # dataset = DataSetMedical("json_data/token_vocab.pkl" , "json_data/label_vocab.pkl" , "json_data/x.txt" , "json_data/y.txt")
    # print(len(dataset))
    # DataLoader_Medical = DataLoader(dataset , batch_size=2 , shuffle=True , drop_last=False)
    # text = '，患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术（DIXON术），手术过程顺利，术后给予抗感染及营养支持治疗，患者恢复好，切口愈合良好。，术后病理示：直肠腺癌（中低度分化），浸润溃疡型，面积3.5*2CM，侵达外膜。双端切线另送“近端”、“远端”及环周底部切除面未查见癌。肠壁一站（10个）、中间组（8个）淋巴结未查见癌。，免疫组化染色示：ERCC1弥漫（+）、TS少部分弱（+）、SYN（-）、CGA（-）。术后查无化疗禁忌后给予3周期化疗，，方案为：奥沙利铂150MG D1，亚叶酸钙0.3G+替加氟1.0G D2-D6，同时给与升白细胞、护肝、止吐、免疫增强治疗，患者副反应轻。院外期间患者一般情况好，无恶心，无腹痛腹胀胀不适，无现患者为行复查及化疗再次来院就诊，门诊以“直肠癌术后”收入院。   近期患者精神可，饮食可，大便正常，小便正常，近期体重无明显变化。'
    # text.strip("")
    # max_len = 30
    # result_segments, result_starts = split_text(text, max_len, split_pat=r'([，。]"?)', greedy=False)
    # for text, start in zip (result_segments,result_starts):
    #     print(text)
    #     print(start)
    params = Param(config = None)
    params.save()
    dataloader = DataLoaderMedical(params)
    train_dataloader = dataloader.get_dataloard("train")
    for batch in train_dataloader:
        print(batch)
    print(" ")


