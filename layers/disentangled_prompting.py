import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptTimeEncoder(nn.Module):
    """
    Encode future timestamps into continuous prompt embeddings.
    """

    def __init__(self, d_model: int, use_weekday: bool = True, use_minute: bool = True):
        super().__init__()
        minute_dim = 10 if use_minute else 0
        hour_dim = 16
        weekday_dim = 8 if use_weekday else 0
        day_dim = 16
        month_dim = 8

        self.use_weekday = use_weekday and weekday_dim > 0
        self.use_minute = use_minute and minute_dim > 0

        self.month_embed = nn.Embedding(13, month_dim)
        self.day_embed = nn.Embedding(32, day_dim)
        self.hour_embed = nn.Embedding(24, hour_dim)
        self.weekday_embed = nn.Embedding(7, weekday_dim) if self.use_weekday else None
        self.minute_embed = nn.Embedding(60, minute_dim) if self.use_minute else None

        concat_dim = month_dim + day_dim + hour_dim + weekday_dim + minute_dim
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    @staticmethod
    def _normalize_shape(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor[:, 0, :]
        return tensor

    def forward(self, time_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        month = self._normalize_shape(time_info['month']).long().clamp(1, 12)
        day = self._normalize_shape(time_info['day']).long().clamp(1, 31)
        hour = self._normalize_shape(time_info['hour']).long().clamp(0, 23)

        embeddings = [
            self.month_embed(month),
            self.day_embed(day),
            self.hour_embed(hour),
        ]

        if self.use_weekday:
            weekday_tensor = time_info.get('weekday')
            if weekday_tensor is None:
                zeros = torch.zeros(
                    month.size(0),
                    month.size(1),
                    self.weekday_embed.embedding_dim,
                    device=month.device,
                )
                embeddings.append(zeros)
            else:
                weekday = self._normalize_shape(weekday_tensor).long().clamp(0, 6)
                embeddings.append(self.weekday_embed(weekday))

        if self.use_minute:
            minute_tensor = time_info.get('minute')
            if minute_tensor is None:
                zeros = torch.zeros(
                    month.size(0),
                    month.size(1),
                    self.minute_embed.embedding_dim,
                    device=month.device,
                )
                embeddings.append(zeros)
            else:
                minute = self._normalize_shape(minute_tensor).long().clamp(0, 59)
                embeddings.append(self.minute_embed(minute))

        return self.proj(torch.cat(embeddings, dim=-1))


class DisentangledPromptCrossAttention(nn.Module):
    """
    Disentangled prompting with trend/season query streams and adaptive image-text fusion.
    """

    def __init__(self, d_model: int, image_dim: int, poi_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.q_base_trend = nn.Parameter(torch.randn(d_model) * 0.02)
        self.q_base_season = nn.Parameter(torch.randn(d_model) * 0.02)
        self.w_trend = nn.Linear(d_model, d_model, bias=False)
        self.w_season = nn.Linear(d_model, d_model, bias=False)

        self.image_k = nn.Linear(image_dim, d_model)
        self.image_v = nn.Linear(image_dim, d_model)
        self.poi_k = nn.Linear(poi_dim, d_model)
        self.poi_v = nn.Linear(poi_dim, d_model)

        self.query_norm = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    @staticmethod
    def _flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
        bsz, num_nodes, token_len, dim = tokens.shape
        return tokens.reshape(bsz * num_nodes, token_len, dim)

    def _cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.query_norm(query)
        k = self.key_norm(key)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits, dim=-1)
        return torch.matmul(attn, value)

    def forward(
        self,
        time_embedding: torch.Tensor,
        image_tokens: torch.Tensor,
        poi_tokens: torch.Tensor,
        poi_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, num_nodes, _, _ = image_tokens.shape
        horizon = time_embedding.size(1)

        q_trend = self.q_base_trend.view(1, 1, -1) + self.w_trend(time_embedding)
        q_season = self.q_base_season.view(1, 1, -1) + self.w_season(time_embedding)

        q_trend_nodes = q_trend.unsqueeze(1).expand(-1, num_nodes, -1, -1).reshape(
            bsz * num_nodes, horizon, self.d_model
        )
        q_season_nodes = q_season.unsqueeze(1).expand(-1, num_nodes, -1, -1).reshape(
            bsz * num_nodes, horizon, self.d_model
        )

        image_flat = self._flatten_tokens(image_tokens)
        image_k = self.image_k(image_flat)
        image_v = self.image_v(image_flat)

        poi_flat = self._flatten_tokens(poi_tokens)
        poi_k = self.poi_k(poi_flat)
        poi_v = self.poi_v(poi_flat)

        poi_mask_flat = None
        if poi_mask is not None:
            poi_mask_flat = poi_mask.reshape(bsz * num_nodes, poi_mask.size(-1)).bool()

        h_img_trend = self._cross_attention(q_trend_nodes, image_k, image_v)
        h_img_season = self._cross_attention(q_season_nodes, image_k, image_v)
        h_poi_trend = self._cross_attention(q_trend_nodes, poi_k, poi_v, poi_mask_flat)
        h_poi_season = self._cross_attention(q_season_nodes, poi_k, poi_v, poi_mask_flat)

        h_img = h_img_trend + h_img_season
        h_poi = h_poi_trend + h_poi_season

        gate_logits = self.gate(torch.cat([h_img, h_poi], dim=-1))
        gates = torch.softmax(gate_logits, dim=-1)

        h_fused = gates[..., :1] * h_img + gates[..., 1:] * h_poi

        def _restore_nodes(x: torch.Tensor) -> torch.Tensor:
            return x.view(bsz, num_nodes, horizon, x.size(-1))

        return {
            'q_trend': q_trend,
            'q_season': q_season,
            'image_dynamic': _restore_nodes(h_img),
            'poi_dynamic': _restore_nodes(h_poi),
            'fused_dynamic': _restore_nodes(h_fused),
            'gates': _restore_nodes(gates),
        }
