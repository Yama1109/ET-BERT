import math
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale=True
    ):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num

        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
        )

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def forward(self, key, value, query, mask, position_bias=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return (
                x.contiguous()
                .view(batch_size, seq_length, heads_num, per_head_size)
                .transpose(1, 2)
            )

        def unshape(x):
            return (
                x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_hidden_size)
            )

        query, key, value = [
            l(x).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        if position_bias is not None:
            scores = scores + position_bias
        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        # asnfm M2/G8: 介入前attentionを観測し、指定時だけkey位置へのmassを縮小して
        # 再正規化する。属性がない場合とscale=1.0の場合は元の計算graphを変更しない。
        self.last_probs_pre_intervention = probs.detach()
        probs_pre_intervention = probs
        scale = getattr(self, "intervention_scale", 1.0)
        positions = getattr(self, "intervention_key_positions", ())
        selected_heads = getattr(self, "intervention_heads", ())
        valid_positions = [position for position in positions if 0 <= position < seq_length]
        if scale != 1.0 and valid_positions:
            factors = torch.ones_like(probs)
            valid_heads = (
                [head for head in selected_heads if 0 <= head < heads_num]
                if selected_heads
                else list(range(heads_num))
            )
            if valid_positions and valid_heads:
                for head in valid_heads:
                    factors[:, head, :, valid_positions] = scale
                probs = probs * factors
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self.last_probs = probs.detach()
        # 分類に使う[CLS] queryから介入対象key位置へ向かうattention massを
        # dataset全体で集約する。明示的に有効化した評価runだけでCPUへ転送する。
        if getattr(self, "collect_attention_stats", False) and valid_positions:
            pre_mass = probs_pre_intervention[:, :, 0, valid_positions].sum(dim=-1)
            post_mass = probs[:, :, 0, valid_positions].sum(dim=-1)
            pre_sum = pre_mass.detach().sum(dim=0).cpu()
            post_sum = post_mass.detach().sum(dim=0).cpu()
            if getattr(self, "attention_pre_mass_sum", None) is None:
                self.attention_pre_mass_sum = pre_sum
                self.attention_post_mass_sum = post_sum
                self.attention_stats_count = batch_size
            else:
                self.attention_pre_mass_sum += pre_sum
                self.attention_post_mass_sum += post_sum
                self.attention_stats_count += batch_size
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output
