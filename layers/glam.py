import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CityGraphContext(nn.Module):
    """
    Lightweight graph attention over spatial nodes to build city-level context.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, knn_k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.knn_k = max(1, knn_k)

        self.q_proj = nn.Linear(feature_dim, hidden_dim)
        self.k_proj = nn.Linear(feature_dim, hidden_dim)
        self.v_proj = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def _build_knn_mask(self, coords: torch.Tensor) -> torch.Tensor:
        bsz, num_nodes, _ = coords.shape
        if num_nodes <= 1:
            return torch.ones(bsz, num_nodes, num_nodes, device=coords.device, dtype=torch.bool)

        k = min(self.knn_k, num_nodes - 1)
        distances = torch.cdist(coords, coords, p=2)
        eye = torch.eye(num_nodes, device=coords.device, dtype=torch.bool).unsqueeze(0)
        distances = distances.masked_fill(eye, float('inf'))

        _, nn_idx = torch.topk(distances, k=k, dim=-1, largest=False)
        mask = torch.zeros(bsz, num_nodes, num_nodes, device=coords.device, dtype=torch.bool)
        mask.scatter_(2, nn_idx, True)
        mask = mask | eye
        return mask

    def forward(self, z: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = self._build_knn_mask(coords)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        node_ctx = self.out_proj(torch.matmul(attn, v))
        node_ctx = self.norm(z + node_ctx)
        global_ctx = node_ctx.mean(dim=1)

        return {
            'node_context': node_ctx,
            'global_context': global_ctx,
            'adj_mask': mask,
        }


class GlobalLocalAdaptiveModulation(nn.Module):
    """
    GLAM: city-level context extraction + FiLM-style node adaptation + prototype routing.
    """

    def __init__(
        self,
        feature_dim: int,
        num_prototypes: int,
        prototype_tau: float = 0.2,
        hidden_dim: int = 128,
        knn_k: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = max(1, num_prototypes)
        self.prototype_tau = max(float(prototype_tau), 1e-4)

        self.city_context = CityGraphContext(
            feature_dim=feature_dim,
            hidden_dim=max(32, hidden_dim),
            knn_k=knn_k,
            dropout=dropout,
        )

        self.gamma_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, feature_dim) * 0.02)

    def forward(self, z: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        graph_states = self.city_context(z, coords)
        node_context = graph_states['node_context']
        global_context = graph_states['global_context']

        gamma = self.gamma_mlp(global_context)
        beta = self.beta_mlp(global_context)
        z_adapted = gamma.unsqueeze(1) * node_context + beta.unsqueeze(1)

        z_norm = F.normalize(z_adapted, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        routing_logits = torch.einsum('bnd,kd->bnk', z_norm, proto_norm) / self.prototype_tau
        routing_weights = torch.softmax(routing_logits, dim=-1)

        return {
            'z_adapted': z_adapted,
            'z_global': global_context,
            'routing_logits': routing_logits,
            'routing_weights': routing_weights,
            'adj_mask': graph_states['adj_mask'],
        }
