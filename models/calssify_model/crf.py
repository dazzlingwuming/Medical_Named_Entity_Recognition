import torch
import torch.nn as nn
from transformers import BertConfig

from untils import Param

# 定义特殊标签（序列开始、序列结束）
START_TAG = "<START>"
END_TAG = "<END>"
torch.manual_seed(42)


def log_sum_exp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """数值稳定的 log_sum_exp 计算（避免指数爆炸/下溢）"""
    max_val, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_tensor = tensor - max_val
    else:
        stable_tensor = tensor - max_val.unsqueeze(dim)
    return max_val + (stable_tensor.exp().sum(dim, keepdim=keepdim)).log()


class CRFLayer(nn.Module):
    def __init__(self, tag2idx):
        super(CRFLayer, self).__init__()
        # transition[i][j] 表示「从标签 j 转移到标签 i」的分数
        self.tag2idx = tag2idx
        tag_size = len(tag2idx)
        self.transition = nn.Parameter(
            torch.randn(tag_size, tag_size),
            requires_grad=True
        )
        # 初始化转移矩阵参数
        self.reset_parameters()

    def reset_parameters(self):
        """重置转移矩阵，并约束 <START>/<END> 相关的不合理转移"""
        nn.init.xavier_normal_(self.transition)
        start_idx = self.tag2idx[START_TAG]
        end_idx = self.tag2idx[END_TAG]
        # 禁止「从任意标签转移到 <START>」（<START> 是序列的隐含起点）
        self.transition.detach()[start_idx, :] = -10000.0
        # 禁止「从 <END> 转移到任意标签」（<END> 是序列的终点）
        self.transition.detach()[:, end_idx] = -10000.0

    def forward(self, feats, mask):
        """
        前向算法：计算「所有可能标签路径」的总分数（带变长序列掩码处理）
        Args:
            feats: 模型对每个位置的标签发射分数，形状为 (seq_len, batch_size, tag_size)
            mask: 序列填充掩码，形状为 (seq_len, batch_size)，1表示有效token，0表示填充
        Return:
            scores: 每个样本的“所有路径总分数”，形状为 (batch_size,)
        """
        seq_len, batch_size, tag_size = feats.size()
        # 初始化前向变量：仅 <START> 位置分数为 0，其余为极小值（log空间下的0和-∞）
        alpha = feats.new_full((batch_size, tag_size), fill_value=-10000.0)
        start_idx = self.tag2idx[START_TAG]
        alpha[:, start_idx] = 0.0

        for t, feat in enumerate(feats):
            # 扩展维度以实现“广播计算”：
            emit_score = feat.unsqueeze(-1)  # (batch_size, tag_size, 1) → 发射分数（与“当前标签”无关，广播到所有前序标签）
            transition_score = self.transition.unsqueeze(0)  # (1, tag_size, tag_size) → 转移分数（与“样本”无关，广播到所有样本）
            alpha_score = alpha.unsqueeze(1)  # (batch_size, 1, tag_size) → 前序标签的累积分数（与“当前标签”无关，广播到所有当前标签）

            # 计算“从所有前序标签转移到当前标签”的分数和
            alpha_score = alpha_score + transition_score + emit_score  # 形状：(batch_size, tag_size, tag_size)

            # 对“前序标签维度”做 log_sum_exp，得到“当前标签”的累积分数
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1) → 当前时间步的掩码
            # 若为有效token（mask_t=1），用新计算的alpha；否则保留上一时间步的alpha
            alpha = log_sum_exp(alpha_score, dim=-1) * mask_t + alpha * torch.logical_not(mask_t)

        # 最后一步：加上“从所有标签转移到 <END>”的分数
        end_idx = self.tag2idx[END_TAG]
        alpha = alpha + self.transition[end_idx].unsqueeze(0)  # 形状：(batch_size, tag_size)

        # 对“最终标签维度”做 log_sum_exp，得到所有路径的总分数
        return log_sum_exp(alpha, dim=-1)  # 形状：(batch_size,)

    def score_sentence(self, feats, tags, mask):
        """计算真实标签序列（黄金路径）的分数
        Args:
            feats: 发射分数，形状为 (seq_len, batch_size, tag_size)
            tags: 真实标签序列，形状为 (seq_len, batch_size)
            mask: 序列填充掩码，形状为 (seq_len, batch_size)，1表示有效token，0表示填充
        Returns:
            scores: 每个样本的黄金路径分数，形状为 (batch_size,)
        """
        seq_len, batch_size, tag_size = feats.size()
        scores = feats.new_zeros(batch_size)  # 初始化分数为0

        # 在标签序列前拼接 <START> 标签，形状变为 (seq_len + 1, batch_size)
        start_tag_idx = self.tag2idx[START_TAG]
        tags = torch.cat([
            tags.new_full((1, batch_size), fill_value=start_tag_idx),
            tags
        ], dim=0)

        # 逐时间步计算“发射分数 + 转移分数”并累加（仅有效位置参与）
        for t, feat in enumerate(feats):
            # 发射分数：取当前时间步每个样本“真实标签”对应的发射分数
            emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[t + 1])])  # (batch_size,)

            # 转移分数：取“前一个标签 → 当前真实标签”的转移分数
            transition_score = torch.stack([
                self.transition[tags[t + 1, b], tags[t, b]]
                for b in range(batch_size)
            ])  # (batch_size,)

            # 仅有效位置（mask[t]=1）累加分数
            scores += (emit_score + transition_score) * mask[t]

        # 加上“最后一个有效标签 → <END> 标签”的转移分数
        end_tag_idx = self.tag2idx[END_TAG]
        transition_to_end = torch.stack([
            self.transition[end_tag_idx, tag[mask[:, b].bool()].long().sum()]
            for b, tag in enumerate(tags.transpose(0, 1))
        ])
        scores += transition_to_end

        return scores

    def viterbi_decode(self, feats, mask):
        """维特比算法，解码最佳路径
        :param feats: 发射分数，形状为 (seq_len, batch_size, tag_size)
        :param mask: 序列填充掩码，形状为 (seq_len, batch_size)（1表示有效token，0表示填充）
        :return best_path: 最优标签序列，形状为 (seq_len, batch_size)
        """
        seq_len, batch_size, tag_size = feats.size()
        # 初始化log空间分数：仅<START>标签分数为0，其余为极小值（近似负无穷）
        scores = feats.new_full((batch_size, tag_size), fill_value=-10000.0)
        start_tag_idx = self.tag2idx[START_TAG]
        scores[:, start_tag_idx] = 0.0
        pointers = []  # 存储每个时间步的“前驱标签指针”

        # 前向遍历每个时间步，计算“到当前标签的最大分数”并记录指针
        for t, feat in enumerate(feats):
            # 广播计算“前序标签分数 + 转移分数”：形状变为 (batch_size, tag_size, tag_size)
            scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)
            # 在“前序标签维度”取最大值，得到“当前标签的最大分数”和“对应的前序标签指针”
            scores_t, pointer = torch.max(scores_t,dim=-1)  # scores_t: (batch_size, tag_size); pointer: (batch_size, tag_size)
            # 加上当前时间步的“发射分数”
            scores_t += feat
            pointers.append(pointer)
            # 掩码处理：有效位置用新分数，填充位置保留原分数
            mask_t = mask[t].unsqueeze(-1)  # 形状 (batch_size, 1)
            scores = scores_t * mask_t + scores * (~mask_t)  # ~mask_t 是逻辑非操作

        # 将指针列表转换为张量，形状为 (seq_len, batch_size, tag_size)
        pointers = torch.stack(pointers, dim=0)
        # 加上“转移到<END>标签”的分数
        end_tag_idx = self.tag2idx[END_TAG]
        scores += self.transition[end_tag_idx].unsqueeze(0)
        # 找到“最终标签”的最大分数及对应的标签索引
        best_score, best_tag = torch.max(scores, dim=-1)  # best_tag: (batch_size,)

        # 回溯阶段：从最终标签反向构建最优路径
        best_path = best_tag.unsqueeze(-1).tolist()  # 初始形状 (batch_size, 1) 的列表
        for i in range(batch_size):
            current_best_tag = best_tag[i]
            valid_len = int(mask[:, i].sum())  # 当前样本的有效序列长度
            # 逆序遍历“有效时间步”的指针，回溯前驱标签
            for ptr_t in reversed(pointers[:valid_len, i]):
                current_best_tag = ptr_t[current_best_tag].item()
                best_path[i].append(current_best_tag)
            # 调整路径（弹出冗余元素 + 反转成正序）
            best_path[i].pop()  # 弹出最后一次冗余的append
            best_path[i].reverse()
        return best_path


if __name__ == '__main__':
    # 初始化配置与参数
    config = BertConfig.from_dict({})  # 空配置（仅为模拟）
    params = Param(config = config)
    crf_layer = CRFLayer(tag2idx=params.label2id_vocab)

    # 生成模拟输入
    feats = torch.rand(10, 2, 21)  # 形状：(序列长度, 批次大小, 标签数量)
    mask = torch.randint(0, 2, (10, 2))  # 随机掩码（0表示填充，1表示有效）

    # 测试“所有路径总分数”计算
    all_paths_score = crf_layer(feats, mask)
    print("所有路径的总分数（每个样本一个值）：\n", all_paths_score)

    # 测试“维特比解码”（最优标签路径）
    best_paths = crf_layer.viterbi_decode(feats, mask)
    print("维特比解码的最优标签路径（形状：序列长度 × 批次大小）：\n", best_paths)
