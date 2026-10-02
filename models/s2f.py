import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embeddings import (
    BackboneInputEmbedding,
    TemporalEmbedding,
    Time2VecTemporalEmbedding,
    SpatialEmbedding,
)
from layers.encoders import FeatureExtractionModule
from layers.disentangled_prompting import (
    PromptTimeEncoder,
    DisentangledPromptCrossAttention,
)
from layers.glam import GlobalLocalAdaptiveModulation


class GraphTemporalExpert(nn.Module):
    """
    Expert head for graph-routed temporal prediction.
    Input: (B, N, T_his, d_model)
    Output: (B, N, T_pred, out_dim)
    """

    def __init__(self, d_model: int, his_len: int, pred_len: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.his_len = his_len
        self.pred_len = pred_len
        self.out_dim = out_dim

        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.out_proj = nn.Linear(d_model, out_dim)
        self.temporal_proj = None
        if his_len != pred_len:
            self.temporal_proj = nn.Linear(his_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, D)
        h = self.token_mlp(x)
        y = self.out_proj(h)
        # y: (B, N, T, out_dim)
        if self.temporal_proj is not None:
            y = y.permute(0, 1, 3, 2)
            y = self.temporal_proj(y)
            y = y.permute(0, 1, 3, 2)
        return y


class S2F(nn.Module):
    """
    Complete Traffic Prediction Model (S2F).
    Compatible with V2 preprocessing caches and dict-based feature payloads.
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_nodes = args.num_nodes
        self.his_len = args.his_len
        self.pred_len = args.pred_len
        self.input_dim = getattr(args, 'input_dim', 2)
        self.d_model = args.d_model
        self.hidden_dim = args.hidden_dim
        self.output_dim = getattr(args, 'output_dim', 2)
        self.prompt_dim = getattr(args, 'prompt_dim', self.hidden_dim)
        self.num_experts = getattr(args, 'num_experts', 8)
        self.expert_topk = max(1, min(getattr(args, 'moe_topk', 2), self.num_experts))
        self.disable_image = bool(getattr(args, 'disable_image', False))
        self.disable_text = bool(getattr(args, 'disable_text', False))
        self.use_glam = bool(getattr(args, 'use_glam', True))
        self.use_prototype_router = bool(getattr(args, 'use_prototype_router', True))

        self.value_embedding = BackboneInputEmbedding(
            c_in=self.input_dim,
            d_model=self.d_model,
            num_nodes=self.num_nodes,
            dropout=args.dropout
        )

        self.feature_extractor = FeatureExtractionModule(
            use_remoteclip=getattr(args, 'use_remoteclip', True),
            remoteclip_model=getattr(args, 'remoteclip_model', 'ViT-B-32'),
            image_token_count=getattr(args, 'image_token_count', 8)
        )

        self.img_dim = self.feature_extractor.img_output_dim
        self.text_dim = self.feature_extractor.text_output_dim

        self.spatial_encoder = SpatialEmbedding(
            coord_dim=2,
            d_model=self.d_model
        )

        self.time_dim = getattr(args, 'feature_dim', 32)
        use_time2vec_temporal = bool(getattr(args, 'use_time2vec_temporal', True))
        if use_time2vec_temporal:
            self.temporal_encoder = Time2VecTemporalEmbedding(
                d_model=self.time_dim,
                use_weekday=True,
                use_minute=True,
                k_sin_features=getattr(args, 'time2vec_k_sin', 63),
            )
        else:
            self.temporal_encoder = TemporalEmbedding(
                d_model=self.time_dim,
                embed_type='learnable',
                use_weekday=True,
                use_minute=True,
            )

        self.mask_flow_input = getattr(args, 'mask_flow_input', True)
        self.temporal_to_backbone = nn.Linear(self.time_dim, self.d_model)

        self.prompt_time_encoder = PromptTimeEncoder(
            d_model=self.prompt_dim,
            use_weekday=True,
            use_minute=True,
        )
        self.disentangled_prompt = DisentangledPromptCrossAttention(
            d_model=self.prompt_dim,
            image_dim=self.img_dim,
            poi_dim=self.text_dim,
            dropout=args.dropout,
        )

        self.image_dyn_proj = nn.Linear(self.prompt_dim, self.d_model)
        self.text_dyn_proj = nn.Linear(self.prompt_dim, self.d_model)

        self.glam_feature_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.glam = GlobalLocalAdaptiveModulation(
            feature_dim=self.hidden_dim,
            num_prototypes=self.num_experts,
            prototype_tau=getattr(args, 'prototype_tau', 0.2),
            hidden_dim=max(self.hidden_dim, 64),
            knn_k=getattr(args, 'glam_knn_k', 8),
            dropout=args.dropout,
        )

        self.temporal_to_dmodel = nn.Linear(self.time_dim, self.d_model)
        self.moe_input_fuse = nn.Sequential(
            nn.LayerNorm(self.d_model * 5),
            nn.Linear(self.d_model * 5, self.d_model),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        self.node_router = nn.Linear(self.hidden_dim, self.num_experts)
        self.experts = nn.ModuleList([
            GraphTemporalExpert(
                d_model=self.d_model,
                his_len=self.his_len,
                pred_len=self.pred_len,
                out_dim=self.output_dim,
                dropout=args.dropout,
            ) for _ in range(self.num_experts)
        ])

        self.freeze_perception_modules()

    @staticmethod
    def _prepare_time_tensor(tensor: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if tensor.dim() == 3:
            if tensor.size(1) == num_nodes:
                return tensor[:, 0, :]
            if tensor.size(2) == num_nodes:
                return tensor[:, :, 0]
        return tensor

    def _extract_future_time_info(self, time_info):
        pred_keys = {
            'month': time_info.get('pred_month'),
            'day': time_info.get('pred_day'),
            'hour': time_info.get('pred_hour'),
            'minute': time_info.get('pred_minute'),
            'weekday': time_info.get('pred_weekday'),
        }
        if pred_keys['month'] is not None and pred_keys['day'] is not None and pred_keys['hour'] is not None:
            return pred_keys

        fallback = {
            'month': time_info['month'],
            'day': time_info['day'],
            'hour': time_info['hour'],
            'minute': time_info.get('minute'),
            'weekday': time_info.get('weekday'),
        }
        return fallback

    def _resize_time_axis(self, x: torch.Tensor, target_t: int) -> torch.Tensor:
        if x.size(2) == target_t:
            return x
        x_t = x.transpose(2, 3)
        x_t = F.interpolate(x_t, size=target_t, mode='nearest')
        return x_t.transpose(2, 3)

    def _match_input_channels(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(-1)
        if c == self.input_dim:
            return x
        if c > self.input_dim:
            return x[..., :self.input_dim]
        pad = torch.zeros(*x.shape[:-1], self.input_dim - c, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    @staticmethod
    def _is_image_embedding_payload(payload) -> bool:
        return isinstance(payload, dict) and ('backbone_features' in payload)

    @staticmethod
    def _is_text_embedding_payload(payload) -> bool:
        return isinstance(payload, dict) and ('backbone_embeddings' in payload)

    def forward(self, X, I, T, coords, time_info, return_gates=False):
        X = self._match_input_channels(X)
        B, N, T_his, _ = X.shape

        h_spatial = self.spatial_encoder(coords)

        # Temporal encoding.
        month_tensor = self._prepare_time_tensor(time_info['month'], self.num_nodes)
        day_tensor = self._prepare_time_tensor(time_info['day'], self.num_nodes)
        hour_tensor = self._prepare_time_tensor(time_info['hour'], self.num_nodes)
        minute_tensor = self._prepare_time_tensor(time_info['minute'], self.num_nodes)
        weekday_tensor = time_info.get('weekday')
        if weekday_tensor is not None:
            weekday_tensor = self._prepare_time_tensor(weekday_tensor, self.num_nodes)

        h_temporal = self.temporal_encoder(
            month=month_tensor,
            day=day_tensor,
            hour=hour_tensor,
            minute=minute_tensor,
            weekday=weekday_tensor
        )

        # Flow embedding.
        if self.mask_flow_input:
            x_emb = torch.zeros(B, N, T_his, self.d_model, device=X.device, dtype=X.dtype)
            spatial_expanded = h_spatial.unsqueeze(2).expand(-1, -1, T_his, -1)
            temporal_proj = self.temporal_to_backbone(h_temporal)
            temporal_expanded = temporal_proj.unsqueeze(1).expand(-1, N, -1, -1)
            x_emb = x_emb + spatial_expanded + temporal_expanded
            x_emb = self.value_embedding.dropout(x_emb)
        else:
            x_emb = self.value_embedding(X)

        h_flow = x_emb

        # Image feature handling (V2 payload).
        if self.disable_image:
            h_img = torch.zeros((B, N, self.img_dim), device=X.device, dtype=h_flow.dtype)
            h_img_tokens = h_img.unsqueeze(2)
        elif self._is_image_embedding_payload(I):
            h_img_backbone = I['backbone_features'].to(device=X.device, dtype=h_flow.dtype)
            h_img = self.feature_extractor.img_proj(h_img_backbone)
            d = h_img.size(-1)
            h_img_tokens = self.feature_extractor.img_token_proj(h_img).view(
                B, N, self.feature_extractor.image_token_count, d
            )
        else:
            raise ValueError('V2 mode requires image backbone feature payload from preprocessed cache')

        # Text feature handling (V2 payload).
        if self.disable_text:
            h_text_all = torch.zeros((B, N, 1, self.text_dim), device=X.device, dtype=h_flow.dtype)
            text_mask = torch.ones(B, N, 1, dtype=torch.bool, device=X.device)
        elif self._is_text_embedding_payload(T):
            h_text_backbone = T['backbone_embeddings'].to(device=X.device, dtype=h_flow.dtype)
            bsz, num_nodes, token_len, dim = h_text_backbone.shape
            h_text_all = self.feature_extractor.text_proj(
                h_text_backbone.view(bsz * num_nodes * token_len, dim)
            ).view(bsz, num_nodes, token_len, dim)
            text_mask = T.get('mask')
            if text_mask is None:
                text_mask = torch.ones(h_text_all.size(0), h_text_all.size(1), h_text_all.size(2),
                                       dtype=torch.bool, device=X.device)
            else:
                text_mask = text_mask.to(device=X.device).bool()
        else:
            raise ValueError('V2 mode requires text backbone embedding payload from preprocessed cache')

        # Disentangled prompting.
        future_time = self._extract_future_time_info(time_info)
        future_time = {
            'month': self._prepare_time_tensor(future_time['month'], self.num_nodes),
            'day': self._prepare_time_tensor(future_time['day'], self.num_nodes),
            'hour': self._prepare_time_tensor(future_time['hour'], self.num_nodes),
            'minute': self._prepare_time_tensor(future_time['minute'], self.num_nodes) if future_time.get('minute') is not None else None,
            'weekday': self._prepare_time_tensor(future_time['weekday'], self.num_nodes) if future_time.get('weekday') is not None else None,
        }
        time_prompt = self.prompt_time_encoder(future_time)
        prompt_states = self.disentangled_prompt(
            time_embedding=time_prompt,
            image_tokens=h_img_tokens,
            poi_tokens=h_text_all,
            poi_mask=text_mask,
        )
        h_img_dyn = self.image_dyn_proj(prompt_states['image_dynamic'])
        h_text_dyn = self.text_dyn_proj(prompt_states['poi_dynamic'])

        target_t = h_flow.size(2)
        h_img_for_moe = self._resize_time_axis(h_img_dyn, target_t)
        h_text_for_moe = self._resize_time_axis(h_text_dyn, target_t)

        temporal_nodes = self.temporal_to_dmodel(h_temporal).unsqueeze(1).expand(-1, N, -1, -1)
        temporal_nodes = self._resize_time_axis(temporal_nodes, target_t)
        spatial_nodes = h_spatial.unsqueeze(2).expand(-1, -1, target_t, -1)

        # Fuse inputs for MoE routing.
        h_moe_in = self.moe_input_fuse(
            torch.cat([h_flow, h_img_for_moe, h_text_for_moe, temporal_nodes, spatial_nodes], dim=-1)
        )

        # Routing and expert aggregation.
        z_local = h_moe_in.mean(dim=2)
        static_descriptor = self.glam_feature_proj(z_local)
        if self.use_glam:
            glam_states = self.glam(static_descriptor, coords)
            prototype_membership = glam_states['routing_weights']
        else:
            glam_states = {}
            prototype_membership = None

        if self.use_prototype_router and prototype_membership is not None:
            route_weights = prototype_membership
        else:
            route_weights = torch.softmax(self.node_router(static_descriptor), dim=-1)

        if self.expert_topk < self.num_experts:
            topk_vals, topk_idx = torch.topk(route_weights, self.expert_topk, dim=-1)
            sparse = torch.zeros_like(route_weights)
            sparse.scatter_(-1, topk_idx, topk_vals)
            route_weights = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)

        expert_outputs = torch.stack([expert(h_moe_in) for expert in self.experts], dim=-1)
        pred = (expert_outputs * route_weights.unsqueeze(2).unsqueeze(2)).sum(dim=-1)
        pred = torch.relu(pred)

        if return_gates:
            router_probs = route_weights.unsqueeze(2).expand(-1, -1, target_t, -1)
            details = {
                'router_probs': router_probs,
                'modality_weights': prompt_states['gates'].mean(dim=2),
                'modality_order': ['image', 'poi'],
                'cluster_membership': route_weights,
                'image_dynamic': h_img_dyn,
                'poi_dynamic': h_text_dyn,
                'fused_dynamic': prompt_states['fused_dynamic'],
                'prompt_gates': prompt_states['gates'],
                'prototype_routing': prototype_membership,
                'h_moe_in': h_moe_in,
            }
            return pred, details
        return pred

    def freeze_perception_modules(self):
        feature_extractor = getattr(self, 'feature_extractor', None)
        if feature_extractor is None:
            return
        freeze_hook = getattr(feature_extractor, 'freeze_pretrained_backbones', None)
        if freeze_hook is not None:
            freeze_hook()