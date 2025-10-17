from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")

print("模型和分词器加载成功")

# from transformers import AlbertTokenizer, AlbertModel
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
# model = AlbertModel.from_pretrained("albert-base-v1")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print("模型和分词器加载成功")
# print(output)\

# from transformers import BertTokenizer, AlbertModel
# tokenizer = BertTokenizer.from_pretrained("uer/albert-base-chinese-cluecorpussmall")
# model = AlbertModel.from_pretrained("uer/albert-base-chinese-cluecorpussmall")
# text = "用你喜欢的任何文本替换我。"
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(output)

# import torch
# from transformers import BertTokenizer, AlbertModel
# tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
# albert = AlbertModel.from_pretrained("clue/albert_chinese_tiny")
# print("模型和分词器加载成功")
