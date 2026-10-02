import math

import torch
import torch.nn as nn


class ValueEmbedding(nn.Module):
    """
    Value embedding using Conv1D (token-style).
    """
    def __init__(self, c_in=1, d_model=64, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='circular',
            bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        B, N, T, C = x.shape
        x = x.view(B * N, T, C).permute(0, 2, 1)
        x = self.tokenConv(x)
        x = x.transpose(1, 2).contiguous().view(B, N, T, -1)
        return x


class TemporalEmbedding(nn.Module):
    """
    Enhanced temporal embedding via concat-then-project.
    """
    def __init__(self, d_model, embed_type='learnable', use_weekday=True, use_minute=True):
        super().__init__()
        self.embed_type = embed_type
        minute_dim = 10 if use_minute else 0
        hour_dim = 16
        weekday_dim = 8 if use_weekday else 0
        day_dim = 16
        month_dim = 8
        self.use_weekday = use_weekday and weekday_dim > 0
        self.use_minute = use_minute and minute_dim > 0
        concat_dim = minute_dim + hour_dim + weekday_dim + day_dim + month_dim
        if concat_dim == 0:
            raise ValueError('TemporalEmbedding requires at least one time feature.')
        self.minute_embed = nn.Embedding(60, minute_dim) if self.use_minute else None
        self.hour_embed = nn.Embedding(24, hour_dim)
        self.weekday_embed = nn.Embedding(7, weekday_dim) if self.use_weekday else None
        self.day_embed = nn.Embedding(32, day_dim)
        self.month_embed = nn.Embedding(13, month_dim)
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, month, day, hour, minute=None, weekday=None):
        B, T = month.shape
        month = month.long().clamp(1, 12)
        day = day.long().clamp(1, 31)
        hour = hour.long().clamp(0, 23)
        embeds = [
            self.month_embed(month),
            self.day_embed(day),
            self.hour_embed(hour)
        ]
        if self.use_weekday:
            if weekday is None:
                embeds.append(torch.zeros(B, T, self.weekday_embed.embedding_dim, device=month.device))
            else:
                weekday = weekday.long().clamp(0, 6)
                embeds.append(self.weekday_embed(weekday))
        if self.use_minute:
            if minute is None:
                embeds.append(torch.zeros(B, T, self.minute_embed.embedding_dim, device=month.device))
            else:
                minute = minute.long().clamp(0, 59)
                embeds.append(self.minute_embed(minute))
        temporal_concat = torch.cat(embeds, dim=-1)
        return self.projection(temporal_concat)


class Time2VecTemporalEmbedding(nn.Module):
    """Temporal embedding using a Time2Vec-style continuous time representation.

    This module shares the same interface and output shape as TemporalEmbedding:
    (month, day, hour[, minute][, weekday]) -> (B, T, d_model), but models time
    as a continuous scalar tau passed through a linear trend term and multiple
    sinusoidal components. This makes the model more robust to small time
    shifts (e.g., 8:15 instead of 8:00) and better at capturing long-range
    periodic structure such as daily/weekly cycles.
    """

    def __init__(
        self,
        d_model: int,
        use_weekday: bool = True,
        use_minute: bool = True,
        k_sin_features: int = 63,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_weekday = use_weekday
        self.use_minute = use_minute
        self.k_sin = int(k_sin_features)

        # Time2Vec core parameters: trend (linear) and seasonal (multiple sines).
        self.w_linear = nn.Parameter(torch.randn(1))
        self.p_linear = nn.Parameter(torch.randn(1))
        self.w_sin = nn.Parameter(torch.randn(self.k_sin))
        self.p_sin = nn.Parameter(torch.randn(self.k_sin))

        # Optional weekday embedding, kept small to act as a categorical bias.
        weekday_dim = 8 if use_weekday else 0
        self.weekday_dim = weekday_dim
        self.weekday_embed = (
            nn.Embedding(7, weekday_dim) if use_weekday and weekday_dim > 0 else None
        )

        in_dim = 1 + self.k_sin + weekday_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        month: torch.Tensor,
        day: torch.Tensor,
        hour: torch.Tensor,
        minute: torch.Tensor = None,
        weekday: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode discrete calendar fields into a continuous Time2Vec embedding.

        All inputs are expected to have shape (B, T). The output has shape
        (B, T, d_model), matching the original TemporalEmbedding, so downstream
        code in Scene2Flow can remain unchanged.
        """

        # Convert to float for continuous computation.
        month_f = month.float()
        day_f = day.float()
        hour_f = hour.float()

        # Pseudo-continuous time tau: accumulated hours within a year-like cycle.
        # This mirrors the PromptTimeEncoder logic so that past and future share
        # a consistent physical time axis.
        tau = (month_f - 1.0) * 30.0 * 24.0 + (day_f - 1.0) * 24.0 + hour_f
        if self.use_minute and (minute is not None):
            minute_f = minute.float()
            tau = tau + minute_f / 60.0

        # (B, T) -> (B, T, 1)
        tau = tau.unsqueeze(-1)

        # Trend and seasonal components.
        v_linear = self.w_linear * tau + self.p_linear              # (B, T, 1)
        v_sin = torch.sin(self.w_sin * tau + self.p_sin)            # (B, T, k_sin)

        feats = [v_linear, v_sin]

        # Optional weekday embedding as an additional categorical bias.
        if self.weekday_dim > 0 and self.weekday_embed is not None:
            if weekday is None:
                B, T, _ = v_linear.shape
                feats.append(
                    torch.zeros(
                        B,
                        T,
                        self.weekday_dim,
                        device=v_linear.device,
                        dtype=v_linear.dtype,
                    )
                )
            else:
                weekday_idx = weekday.long().clamp(0, 6)
                feats.append(self.weekday_embed(weekday_idx))

        time_feats = torch.cat(feats, dim=-1)                        # (B, T, in_dim)
        return self.projection(time_feats)                            # (B, T, d_model)


class SpatialEmbedding(nn.Module):
    """
    Spatial encoding with Gaussian Fourier Features.
    """
    def __init__(self, coord_dim=2, d_model=64, scale=10.0, learnable=True):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError('d_model must be even for Gaussian Fourier features.')
        self.coord_dim = coord_dim
        b_matrix = torch.randn(coord_dim, d_model // 2) * scale
        if learnable:
            self.b_matrix = nn.Parameter(b_matrix)
        else:
            self.register_buffer('b_matrix', b_matrix)
        norm = torch.ones(coord_dim)
        if coord_dim >= 1:
            norm[0] = 180.0
        if coord_dim >= 2:
            norm[1] = 90.0
        self.register_buffer('norm_scale', norm)

    def forward(self, coords):
        scale = self.norm_scale.to(coords.device, coords.dtype).view(1, 1, -1)
        coords_norm = coords / scale
        proj = coords_norm @ self.b_matrix
        proj = 2 * math.pi * proj
        fourier_features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return fourier_features


class SpatioTemporalEmbedding(nn.Module):
    """
    Spatial + temporal encoding using node IDs.
    """
    def __init__(self, d_model, num_nodes, max_len=5000):
        super().__init__()
        if num_nodes is None:
            raise ValueError('SpatioTemporalEmbedding requires num_nodes for node-wise encoding.')
        self.num_nodes = num_nodes
        self.spatial_embedding = nn.Embedding(num_nodes, d_model)
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('temporal_embedding', pe)

    def forward(self, B, N, T, device):
        node_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        spatial = self.spatial_embedding(node_ids).unsqueeze(2)
        temporal = self.temporal_embedding[:T, :].to(device).unsqueeze(0).unsqueeze(0)
        spatio_temporal = spatial + temporal
        return spatio_temporal


class BackboneInputEmbedding(nn.Module):
    """
    Backbone input embedding with spatio-temporal positional cues.
    """
    def __init__(self, c_in, d_model, num_nodes, dropout=0.1):
        super().__init__()
        self.value_embedding = ValueEmbedding(c_in, d_model)
        self.position_embedding = SpatioTemporalEmbedding(d_model, num_nodes)
        self.dropout = nn.Dropout(p=dropout)
        self.use_position = True

    def get_positional_encoding(self, B, N, T, device):
        return self.position_embedding(B, N, T, device)

    def forward(self, x):
        B, N, T, _ = x.shape
        value_emb = self.value_embedding(x)
        pos_emb = self.position_embedding(B, N, T, x.device)
        return self.dropout(value_emb + pos_emb)