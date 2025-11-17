# Medical_Named_Entity_Recognition

## 项目简介
本项目专注于医疗领域的命名实体识别（Named Entity Recognition, NER）任务，旨在从医疗文本中自动识别出特定实体（如疾病类型等）。项目采用BIOES标签体系对实体进行标注，结合多种自然语言处理模型结构，实现高效、准确的实体识别，可应用于医疗文本分析、电子病历处理等场景。


## 环境要求
- Python 3.7+
- 依赖库：`json`（数据读取）、`pandas`（数据处理，可选）、`numpy`（数值计算）、`torch`（模型训练与推理）、`transformers`（预训练模型加载，如BERT）等


## 数据说明
### 数据格式
- 原始数据以`JSON`格式存储，每条数据包含`originalText`（原始文本）和`entities`（实体标签信息）。
- 标签体系采用**BIOES**：
  - `B-实体类型`：实体的开始位置
  - `I-实体类型`：实体的中间位置
  - `O`：非实体
  - `E-实体类型`：实体的结束位置
  - `S-实体类型`：单个字符的实体

### 数据处理
数据处理核心逻辑位于`data`目录下：
- `dataloard_utils.py`：提供工具函数，如`read_json`用于读取JSON格式数据。
- `creat_data.py`：实现数据划分与处理，通过`split_data_2_x_y`函数将原始数据拆分为文本序列`x`和标签序列`y`，并支持生成对比文件用于数据校验。
- 特殊字符处理：`json_data/unique_chars.txt`包含数据中出现的独特字符，辅助词汇表构建与异常值处理。

### 注意事项
- 长文本需进行分片处理，避免拆分实体（需确保分片逻辑不破坏实体的完整性）。
- 以字为单位进行处理，可结合`char2vec`初始化字向量。


## 代码结构
```
Medical_Named_Entity_Recognition/
├── data/                      # 数据处理相关
│   ├── dataloard_utils.py     # 数据工具函数（如JSON读取）
│   ├── creat_data.py          # 数据划分与处理
│   ├── json_data/             # 数据存储目录
│   │   ├── 对比.txt           # 数据校验文件
│   │   └── unique_chars.txt   # 独特字符集
├── pre_train_model/           # 预训练模型相关
│   └── bert/
│       └── vocab.txt          # BERT词汇表
└── 笔记.txt                   # 项目设计思路与说明
```


## 模型构建
### 整体结构
模型采用三层架构：`Embedding层 -> Encoder层 -> Classifier层`

1. **Embedding层**  
   输入：文本序列的`input_ids`和`input_mask`（形状：`[batch_size, seq_len]`）  
   输出：文本的向量表示（形状：`[batch_size, seq_len, embedding_dim]`）  
   支持的预训练模型：`Embedding/ELMo/BERT/NEZHA`等，可选择冻结参数直接使用或微调（建议使用小学习率）。

2. **Encoder层**  
   输入：Embedding层输出（`[batch_size, seq_len, embedding_dim]`）  
   输出：特征提取结果（`[batch_size, seq_len, hidden_size*2]`）  
   可选模型及特点：
   | 模型               | 方向信息 | 相对距离信息 | 局部信息 | 长距离依赖 | 可并行性 |
   |--------------------|----------|--------------|----------|------------|----------|
   | BILSTM             | 高       | 高           | 差       | 中         | 差       |
   | ID-CNN             | 差       | 差           | 高       | 差         | 高       |
   | Transformer        | 差       | 差           | 差       | 高         | 高       |
   | R-Transformer      | 中-高    | 中-高        | 高       | 高         | 中       |

3. **Classifier层**  
   输入：Encoder层输出（`[batch_size, seq_len, hidden_size*2]`）  
   输出：实体标签预测结果（`[batch_size, seq_len, num_labels]`）  
   可选模型：全连接网络（FC）或条件随机场（CRF）。


### 模型选择建议
- 基础模型：`word2vec -> BILSTM -> FC`  
- 进阶模型：`BERT/NEZHA (Embedding) -> Transformer/R-Transformer (Encoder) -> FC/CRF (Classifier)`


## 使用方法
1. **数据准备**  
   将原始JSON格式数据放入`data/json_data/`目录。

2. **数据处理**  
   运行`creat_data.py`生成训练所需的文本序列`x`和标签序列`y`：
   ```bash
   python data/creat_data.py
   ```
   处理结果可通过`json_data/对比.txt`查看文本与标签的对应关系。

3. **模型训练与推理**  
   （需补充模型训练脚本，如`train.py`）  
   训练：
   ```bash
   python train.py --model bert --encoder transformer --classifier crf
   ```
   推理：
   ```bash
   python predict.py --input "待识别的医疗文本"
   ```


## 注意事项
- 预训练模型（如BERT）需确保`vocab.txt`与数据字符集匹配，避免未登录词问题。
- 长文本分片时需自定义逻辑，优先保证实体不被拆分。
- 训练时可根据数据规模调整预训练模型的微调策略（冻结/解冻参数）。
