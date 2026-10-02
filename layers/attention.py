import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionPooling(nn.Module):
    """
    Attention-based pooling mechanism.
    Use query features to dynamically aggregate value features.
    """
    def __init__(self, query_dim, value_dim, hidden_dim=None):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim or value_dim

        self.query_proj = nn.Linear(query_dim, value_dim)
        self.scale = math.sqrt(value_dim)

    def forward(self, query, values, attention_mask=None):
        B, N, L, D_val = values.shape

        query_proj = self.query_proj(query)
        query_proj = query_proj.unsqueeze(2)

        query_norm = F.normalize(query_proj, p=2, dim=-1)
        values_norm = F.normalize(values, p=2, dim=-1)

        attention_scores = torch.matmul(query_norm, values_norm.transpose(-2, -1))
        attention_scores = attention_scores / self.scale

        if attention_mask is not None:
            attention_mask_exp = attention_mask.unsqueeze(2).float()
            mask_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill(
                attention_mask_exp == 0, mask_value
            )

        attention_weights = F.softmax(attention_scores, dim=-1)

        pooled = torch.matmul(attention_weights, values)
        pooled = pooled.squeeze(2)

        return pooled


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention layer.
    Can be used for various attention mechanisms in the network.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        B, L, _ = query.shape

        Q = self.q_linear(query).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, mask_value)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention, V)

        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)

        output = self.out_linear(output)

        return output, attention