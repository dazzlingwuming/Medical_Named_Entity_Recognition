import torch
import numpy as np
from models.LM_model.word2vec import Word2vecLMModel
from untils import Param

if __name__ == "__main__":
    params = {}
    params = Param(params=params)
    lab_vocab = torch.load(params.label2id)
    print(lab_vocab)