# -*- coding: utf-8 -*-
import json
import os.path
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BertTokenizer
from transformers import BertConfig
from untils import Param

def split_text_with_entities(text, max_len, split_pat=r'([，。]"?)|([.!?]"?)|(\s+)', greedy=False):
    """
    文本分片，考虑实体边界避免分割实体

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

    # 查找可能的实体边界（基于标点和空格）
    boundaries = []

    # 查找所有标点符号位置
    for match in re.finditer(split_pat, text):
        boundaries.append(match.start())

    # 添加文本开头和结尾作为边界
    boundaries = [0] + sorted(set(boundaries)) + [len(text)]

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

        # 在当前片段范围内查找合适的分割点
        # 优先选择实体边界（标点符号）
        best_boundary = -1

        # 从后往前查找边界点
        for i in range(len(boundaries) - 1, -1, -1):
            if boundaries[i] <= end and boundaries[i] > start:
                # 检查这个边界是否合适
                segment_length = boundaries[i] - start
                if segment_length <= max_len:
                    best_boundary = boundaries[i]
                    break

        # 如果没有找到合适的边界，尝试在空格处分割
        if best_boundary == -1:
            # 查找最后一个空格
            segment = text[start:end]
            last_space = segment.rfind(' ')
            if last_space != -1:
                best_boundary = start + last_space + 1  # 包含空格

        # 如果仍然没有找到合适的分割点，就在max_len处强制分割
        if best_boundary == -1:
            best_boundary = end

        # 添加片段
        segments.append(text[start:best_boundary])
        starts.append(start)

        # 更新起始位置
        start = best_boundary

    return segments, starts

#读取json文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

#读取txt文件
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

class InputExamples(object):
    '''
    一条样本包含的token set ，tag set
    '''
    def __init__(self, sentence, label):
        super(InputExamples, self).__init__()
        self.sentence = sentence
        self.label = label

class InputFeatures(object):
    '''
    一条样本的特征
    '''
    def __init__(self,example_ids,input_ids ,label_ids ,input_mask , split_to_original_id):
        super(InputFeatures, self).__init__()
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.input_mask = input_mask
        #样本分割
        self.example_ids = example_ids#文本id
        self.split_to_original_id = split_to_original_id
    pass

# 读取样本
def read_examples(data_x_file, data_y_file):
    '''
    读取样本
    :param data_x_file:
    :param data_y_file:
    :return:examples:[list] InputExamples
    '''
    examples = []
    with open(data_x_file, "r", encoding="utf-8") as f, open(data_y_file, "r", encoding="utf-8") as g:
        x_data = read_txt(data_x_file)
        y_data = read_txt(data_y_file)
        assert len(x_data) == len(y_data)  ,"长度不一致" #保证x和y的长度一致
        for i in range(len(x_data)):
            x = x_data[i].strip()
            y = y_data[i].strip().split(" ")
            assert len(x) == len(y) , f"第{i}行长度不一致"
            examples.append(InputExamples(sentence=x, label=y))
    print("examples的长度为：", len(examples))
    return examples

# 将样本转换为特征
def convert_examples_to_features(examples, word2id, tag2id, max_seq_length, tokenizer:PreTrainedTokenizer,data_sign="train" ,pad_sign = True , pad_token="[PAD]"):
    '''
    将样本转换为特征
    :param examples:  原始样本
    :param word2id: x的词典
    :param tag2id: y的词典
    :param max_seq_length:最大长度
    :param tokenizer:
    :param data_sign:
    :return: InputFeatures
    '''
    features = []
    pad_token = tokenizer.tokenize(pad_token)[0] if len(tokenizer.tokenize(pad_token)) == 1 else '[UNK]'
    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    for (example_ids, example) in tqdm(enumerate(examples)):
        #对长文本进行分片
        original_id = list(range(len(example.sentence)))
        subtexts , strats = split_text_with_entities(example.sentence, max_seq_length)

        #获取每一个sub_text对应的InputFeatures
        for subtexts_text , start in zip(subtexts, strats):
            text_tokens = [tokenizer.tokenize(token)[0] if len(tokenizer.tokenize(token)) ==1 else '[UNK]' for token in subtexts_text]
        #获取对应的标签ID
            label_ids = example.label[start : start + len(subtexts_text)]
            label_ids= [tag2id[label] if label in tag2id else -1 for label in label_ids] #如果标签不在tag2id中，则标记为"-1"
        #原始文本中的位置信息
            split_to_original_id = original_id[start:start + len(text_tokens)]
            assert len(text_tokens) == len(label_ids) == len(split_to_original_id) , f"第{example_ids}个样本的长度不一致"
            #截断：如果长度大于max_seq_length，则截断
            if len(text_tokens) > max_seq_length:
                text_tokens = text_tokens[:max_seq_length]
                label_ids = label_ids[:max_seq_length]
                split_to_original_id = split_to_original_id[:max_seq_length]
            #token转id
            input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
            assert len(input_ids) == len(text_tokens) , f"第{example_ids}个样本的text长度不一致"
            assert  len(input_ids) == len(label_ids) , f"第{example_ids}个样本的label长度不一致"
            #填充
            if pad_sign and len(input_ids) < max_seq_length:
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([pad_id] * padding_length)
                label_ids = label_ids + ([tag2id["O"]] * padding_length)
                split_to_original_id = split_to_original_id + ([-1] * padding_length)
            #mask
            input_mask = [1 if id != pad_id else 0 for id in input_ids]

            #构建InputFeatures
            features.append(InputFeatures(example_ids = example_ids , input_ids = input_ids ,
                                          label_ids = label_ids , input_mask = input_mask ,
                                          split_to_original_id = split_to_original_id
                                          ))
    return features


if __name__ == "__main__":
    #测试文本分片
    # examples = read_examples("json_data/x.txt", "json_data/y.txt")
    # # text = "，患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术（DIXON术），手术过程顺利，术后给予抗感染及营养支持治疗，患者恢复好，切口愈合良好。，术后病理示：直肠腺癌（中低度分化），浸润溃疡型，面积3.5*2CM，侵达外膜。双端切线另送“近端”、“远端”及环周底部切除面未查见癌。肠壁一站（10个）、中间组（8个）淋巴结未查见癌。，免疫组化染色示：ERCC1弥漫（+）、TS少部分弱（+）、SYN（-）、CGA（-）。术后查无化疗禁忌后给予3周期化疗，，方案为：奥沙利铂150MG D1，亚叶酸钙0.3G+替加氟1.0G D2-D6，同时给与升白细胞、护肝、止吐、免疫增强治疗，患者副反应轻。院外期间患者一般情况好，无恶心，无腹痛腹胀胀不适，无现患者为行复查及化疗再次来院就诊，门诊以“直肠癌术后”收入院。   近期患者精神可，饮食可，大便正常，小便正常，近期体重无明显变化。"
    # # max_len = 20
    # # segments, starts = split_text_with_entities(text, max_len)
    # # print(starts)
    # # print(segments)
    # word2id={}
    # label2id = torch.load("json_data/label_vocab.pkl")
    # max_len = 128
    # tokenizer = BertTokenizer(vocab_file="json_data/unique_chars.txt")
    # feature = convert_examples_to_features(examples, word2id = word2id , tag2id= label2id ,tokenizer=tokenizer, max_seq_length=max_len ,
    #                              data_sign="train" , pad_sign=True , pad_token="[PAD]")
    # print(len(feature))
    # print(len(examples))
    # params = {}
    # root_path = Path(params.get("root_path", os.path.abspath(os.path.dirname(__file__))))
    params = Param()
    params.save()
    pass

