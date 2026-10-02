import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.s2f import S2F
from utils.cached_dataloader import load_cached_datasets
from utils.metrics import compute_metrics


def get_args():
    """
    Parse command line arguments for training.
    """
    parser = argparse.ArgumentParser(description='Scene2Flow training (modality-level MoE)')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (e.g., config/BJ.conf)')

    parser.add_argument('--data', type=str, default='NYC',
                        choices=['BM', 'DC', 'NYC'])
    parser.add_argument('--num_nodes', type=int, default=127)
    parser.add_argument('--his_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=2,
                        help='Flow input channel count (e.g., in/out = 2)')
    parser.add_argument('--output_dim', type=int, default=2,
                        help='Prediction channel count (e.g., in/out = 2)')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--moe_topk', type=int, default=2,
                        help='Top-k experts per modality token (default: 2)')
    parser.add_argument('--feature_dim', type=int, default=32)
    parser.add_argument('--prompt_dim', type=int, default=128,
                        help='Disentangled prompting hidden size')
    parser.add_argument('--image_token_count', type=int, default=8,
                        help='Number of pseudo image tokens expanded from static image feature')
    parser.add_argument('--use_remoteclip', action='store_true', default=True)
    parser.add_argument('--remoteclip_model', type=str, default='ViT-B-32',
                        choices=['RN50', 'ViT-B-32', 'ViT-L-14'])
    parser.add_argument('--use_glam', action='store_true', default=True,
                        help='Enable Global-Local Adaptive Modulation before MoE')
    parser.add_argument('--glam_knn_k', type=int, default=8,
                        help='Neighborhood size for city-level graph context')
    parser.add_argument('--prototype_tau', type=float, default=0.2,
                        help='Temperature for prototype routing softmax')
    parser.add_argument('--use_prototype_router', action='store_true', default=True,
                        help='Use prototype-anchored routing weights as MoE membership')
    parser.add_argument('--mask_flow_input', action='store_true', default=True,
                        help='Zero out historical flow inputs and rely on priors')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--lambda_pred', type=float, default=2.0,
                        help='Weight for the main MSE prediction term (default: 2.0)')
    parser.add_argument('--lambda_amp', type=float, default=0.0,
                        help='Weight for mean-amplitude alignment (disabled by default)')
    parser.add_argument('--lambda_scale', type=float, default=0.0,
                        help='Weight for scale-consistency loss (disabled by default)')
    parser.add_argument('--lambda_msle', type=float, default=5.0,
                        help='Weight for MSLE loss, which is sensitive to magnitude')
    parser.add_argument('--lambda_rank', type=float, default=0.5,
                        help='Weight for spatial ranking loss over node pairs')
    parser.add_argument('--lambda_balance', type=float, default=0.1,
                        help='Weight for cluster load-balancing loss to reduce gating skew (default: 0.1)')
    parser.add_argument('--lambda_tcl', type=float, default=0.1,
                        help='Weight for cross-modal temporal contrastive loss')
    parser.add_argument('--lambda_orth', type=float, default=0.05,
                        help='Weight for cross-modal feature orthogonality loss')
    parser.add_argument('--contrastive_tau', type=float, default=0.2,
                        help='InfoNCE temperature for temporal contrastive alignment')
    parser.add_argument('--orth_eps', type=float, default=1e-6,
                        help='Stability epsilon for orthogonality normalization')
    parser.add_argument('--rank_margin', type=float, default=0.05,
                        help='Threshold for ranking loss (0.05 is recommended for 0-1 normalized data)')
    parser.add_argument('--scale_epsilon', type=float, default=1.0,
                        help='Stability epsilon in the scale-loss denominator to avoid division by zero')
    parser.add_argument('--primary_metric', type=str, default='rmsle',
                        choices=['amp', 'mae', 'mape', 'rmsle', 'composite'],
                        help='Primary metric used to select the best model; RMSLE by default (lower is better)')
    parser.add_argument('--composite_amp_weight', type=float, default=1.0,
                        help='Amplitude weight used when primary_metric=composite')
    parser.add_argument('--mape_save_delta', type=float, default=0.1,
                        help='When primary_metric=mape, save only if MAPE improves by at least this margin')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of epochs for linear learning-rate warmup')
    parser.add_argument('--cosine_tmax_multiplier', type=float, default=1.5,
                        help='Sets CosineAnnealing T_max = epochs * this multiplier')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--save_dir', type=str, default='./checkpoints/')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--cluster_log_interval', type=int, default=None,
                        help='Logging interval for cluster-usage monitoring, in batches')
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    if args.config:
        from utils.config_loader import load_config, config_to_args
        config = load_config(args.config)
        config_args = config_to_args(config)
        for key, value in vars(config_args).items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    if args.cluster_log_interval is None:
        args.cluster_log_interval = max(1, int(args.log_interval))

    return args


def _to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_device(v, device) for v in data)
    return data


def _resolve_default_cache_dir(data_name: str) -> str:
    return f'preprocessed_data/{data_name}'


def _amplitude_loss(pred, target):
    amp_pred = pred.mean(dim=-1)
    amp_true = target.mean(dim=-1)
    return torch.mean(torch.abs(amp_pred - amp_true))


def _scale_loss(pred, target, eps):
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps))


def _msle_loss(pred, target):
    pred_clamped = torch.clamp(pred, min=0.0)
    target_clamped = torch.clamp(target, min=0.0)
    return torch.mean((torch.log1p(pred_clamped) - torch.log1p(target_clamped)) ** 2)


def _temporal_contrastive_loss(image_dynamic, poi_dynamic, coords, knn_k: int, tau: float):

    if image_dynamic is None or poi_dynamic is None or coords is None:
        return None

    if image_dynamic.dim() == 4:
        image_feat = image_dynamic.mean(dim=2)
    else:
        image_feat = image_dynamic
        
    if poi_dynamic.dim() == 4:
        poi_feat = poi_dynamic.mean(dim=2)
    else:
        poi_feat = poi_dynamic

    if image_feat.size(0) == 0 or poi_feat.size(0) == 0 or image_feat.size(1) == 0:
        return None

    bsz, num_nodes, _ = coords.shape
    k = min(knn_k, num_nodes - 1)
    if k > 0:
        distances = torch.cdist(coords, coords, p=2)
        eye = torch.eye(num_nodes, device=coords.device, dtype=torch.bool).unsqueeze(0)
        distances = distances.masked_fill(eye, float('inf'))
        
        _, nn_idx = torch.topk(distances, k=k, dim=-1, largest=False)
        adj_matrix = torch.zeros(bsz, num_nodes, num_nodes, device=coords.device, dtype=torch.bool)
        adj_matrix.scatter_(2, nn_idx, True)
        
        adj_matrix = adj_matrix | eye 
    else:
        adj_matrix = torch.eye(num_nodes, device=coords.device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)

    image_feat = F.normalize(image_feat, dim=-1)
    poi_feat = F.normalize(poi_feat, dim=-1)

    logits = torch.bmm(image_feat, poi_feat.transpose(1, 2)) / max(float(tau), 1e-6)

    logits_max_i2t, _ = torch.max(logits, dim=2, keepdim=True)
    logits_max_t2i, _ = torch.max(logits.transpose(1, 2), dim=2, keepdim=True)
    
    exp_logits_i2t = torch.exp(logits - logits_max_i2t)
    exp_logits_t2i = torch.exp(logits.transpose(1, 2) - logits_max_t2i)

    pos_sum_i2t = (exp_logits_i2t * adj_matrix).sum(dim=2)
    all_sum_i2t = exp_logits_i2t.sum(dim=2)
    
    pos_sum_t2i = (exp_logits_t2i * adj_matrix.transpose(1, 2)).sum(dim=2)
    all_sum_t2i = exp_logits_t2i.sum(dim=2)

    loss_i2t = -torch.log(pos_sum_i2t / (all_sum_i2t + 1e-8)).mean()
    loss_t2i = -torch.log(pos_sum_t2i / (all_sum_t2i + 1e-8)).mean()

    return 0.5 * (loss_i2t + loss_t2i)


def _orthogonality_loss(image_dynamic, poi_dynamic, eps: float):
    if image_dynamic is None or poi_dynamic is None:
        return None

    img = image_dynamic.reshape(-1, image_dynamic.size(-1))
    poi = poi_dynamic.reshape(-1, poi_dynamic.size(-1))
    if img.size(0) == 0 or poi.size(0) == 0:
        return None

    if img.size(0) != poi.size(0):
        shared = min(img.size(0), poi.size(0))
        img = img[:shared]
        poi = poi[:shared]

    img = img - img.mean(dim=0, keepdim=True)
    poi = poi - poi.mean(dim=0, keepdim=True)
    img = img / (img.norm(dim=0, keepdim=True) + eps)
    poi = poi / (poi.norm(dim=0, keepdim=True) + eps)

    corr = torch.matmul(img.transpose(0, 1), poi)
    corr = corr / max(float(img.size(0)), 1.0)
    return torch.mean(corr.pow(2))


def _load_balance_loss(membership: torch.Tensor) -> torch.Tensor:
    """
    Encourage average cluster usage to approach a uniform distribution.
    """
    if membership is None:
        raise ValueError('Cluster membership tensor is required for load balance loss')
    if membership.dim() != 3:
        raise ValueError(f'Expected membership shape (B, N, K), got {membership.shape}')
    mean_usage = membership.mean(dim=(0, 1))
    target = torch.ones_like(mean_usage) / mean_usage.size(0)
    return F.mse_loss(mean_usage, target)


def _zero_scalar_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=tensor.device, dtype=tensor.dtype)


def _align_prediction_target(pred: torch.Tensor, target: torch.Tensor):
    """
    Align prediction/target shapes for scalar decoder outputs against multivariate labels.
    """
    while pred.dim() > target.dim() and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    while target.dim() > pred.dim() and target.size(-1) == 1:
        target = target.squeeze(-1)

    if pred.dim() == 3 and target.dim() == 4:
        if target.size(-1) > 1 and not getattr(_align_prediction_target, '_warned', False):
            print(
                f"[WARN] Target has {target.size(-1)} feature channels while decoder outputs scalar values. "
                "Collapsing target feature channels by mean for loss/metrics alignment."
            )
            _align_prediction_target._warned = True
        target = target.mean(dim=-1)
    elif pred.dim() == 4 and target.dim() == 3:
        pred = pred.mean(dim=-1)
    elif pred.dim() == 4 and target.dim() == 4 and pred.size(-1) != target.size(-1):
        pred = pred.mean(dim=-1)
        target = target.mean(dim=-1)

    return pred, target


def _ranking_loss(pred, target, threshold, rank_loss_fn):
    if pred.dim() > 2:
        val_pred = pred.mean(dim=-1).reshape(-1)
        val_true = target.mean(dim=-1).reshape(-1)
    else:
        val_pred = pred.reshape(-1)
        val_true = target.reshape(-1)

    if val_pred.numel() < 2:
        return pred.new_tensor(0.0)

    indices = torch.randperm(val_pred.size(0), device=val_pred.device)
    pred1 = val_pred
    pred2 = val_pred[indices]
    true1 = val_true
    true2 = val_true[indices]

    diff_abs = torch.abs(true1 - true2)
    thr = max(float(threshold), 0.0)
    diff_mask = diff_abs > thr if thr > 0 else diff_abs > 0
    if diff_mask.sum() == 0 and diff_abs.numel() > 0:
        topk = max(1, diff_abs.numel() // 10)
        _, top_idx = torch.topk(diff_abs, topk)
        diff_mask = torch.zeros_like(diff_abs, dtype=torch.bool)
        diff_mask[top_idx] = True
    if diff_mask.sum() == 0:
        return pred.new_tensor(0.0)

    target_sign = torch.sign(true1 - true2)
    target_sign[target_sign == 0] = 1.0
    return rank_loss_fn(pred1[diff_mask], pred2[diff_mask], target_sign[diff_mask])


def _primary_score(args, metrics):
    if args.primary_metric == 'amp':
        return metrics['AMP_MAE']
    if args.primary_metric == 'mae':
        return metrics['MAE']
    if args.primary_metric == 'mape':
        return metrics['MAPE']
    if args.primary_metric == 'rmsle':
        return metrics.get('RMSLE', float('inf'))
    return metrics['MAE'] + args.composite_amp_weight * metrics['AMP_MAE']


def _step_scheduler_with_warmup(optimizer, scheduler, args, epoch):
    if scheduler is None:
        return
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        warmup_lr = args.lr * float(epoch + 1) / args.warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    else:
        scheduler.step()


def train_epoch(model, train_loader, optimizer, criterion, rank_loss_fn, args, epoch):
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_amp_loss = 0.0
    total_scale_loss = 0.0
    total_msle_loss = 0.0
    total_rank_loss = 0.0
    total_weighted_pred = 0.0
    total_weighted_amp = 0.0
    total_weighted_scale = 0.0
    total_weighted_msle = 0.0
    total_weighted_rank = 0.0
    total_balance_loss = 0.0
    total_weighted_balance = 0.0
    total_tcl_loss = 0.0
    total_orth_loss = 0.0
    total_weighted_tcl = 0.0
    total_weighted_orth = 0.0
    cluster_usage_sum = None
    cluster_usage_steps = 0

    log_every = max(1, int(getattr(args, 'log_interval', 10)))
    pbar = tqdm(
        train_loader,
        desc=f'Epoch {epoch + 1}/{args.epochs} [Train]',
        miniters=log_every,
        mininterval=1.0,
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(pbar):
        img, text, x, y, coords, time_info = batch

        img = _to_device(img, args.device)
        text = _to_device(text, args.device)
        x = x.to(args.device)
        y = y.to(args.device)
        coords = coords.to(args.device)
        time_info = _to_device(time_info, args.device)

        if x.dim() == 3:
            x = x.unsqueeze(-1)

        optimizer.zero_grad()
        pred, routing_details = model(
            X=x,
            I=img,
            T=text,
            coords=coords,
            time_info=time_info,
            return_gates=True
        )
        membership = None
        if isinstance(routing_details, dict):
            membership = routing_details.get('cluster_membership')
        if membership is None:
            raise RuntimeError('Cluster membership is missing despite return_gates=True')
        cluster_usage = membership.mean(dim=(0, 1))
        if cluster_usage_sum is None:
            cluster_usage_sum = torch.zeros_like(cluster_usage)
        cluster_usage_sum += cluster_usage.detach()
        cluster_usage_steps += 1

        pred, y = _align_prediction_target(pred, y)

        pred_loss = criterion(pred, y)

        if args.lambda_amp > 0:
            amp_for_loss = _amplitude_loss(pred, y)
            amp_term = args.lambda_amp * amp_for_loss
            amp_metric = amp_for_loss.detach()
        else:
            amp_term = _zero_scalar_like(pred_loss)
            with torch.no_grad():
                amp_metric = _amplitude_loss(pred.detach(), y.detach())

        if args.lambda_scale > 0:
            scale_for_loss = _scale_loss(pred, y, args.scale_epsilon)
            scale_term = args.lambda_scale * scale_for_loss
            scale_metric = scale_for_loss.detach()
        else:
            scale_term = _zero_scalar_like(pred_loss)
            with torch.no_grad():
                scale_metric = _scale_loss(pred.detach(), y.detach(), args.scale_epsilon)

        msle_loss = _msle_loss(pred, y)
        rank_loss = _ranking_loss(pred, y, args.rank_margin, rank_loss_fn)
        image_dynamic = routing_details.get('image_dynamic') if isinstance(routing_details, dict) else None
        poi_dynamic = routing_details.get('poi_dynamic') if isinstance(routing_details, dict) else None
        tcl_loss = _temporal_contrastive_loss(
            image_dynamic, 
            poi_dynamic, 
            coords,                 
            args.glam_knn_k,        
            args.contrastive_tau
        )
        orth_loss = _orthogonality_loss(image_dynamic, poi_dynamic, args.orth_eps)
        if tcl_loss is None:
            tcl_loss = _zero_scalar_like(pred_loss)
        if orth_loss is None:
            orth_loss = _zero_scalar_like(pred_loss)

        if args.lambda_balance > 0:
            lb_loss = _load_balance_loss(membership)
            balance_term = args.lambda_balance * lb_loss
        else:
            lb_loss = _zero_scalar_like(pred_loss)
            balance_term = lb_loss
        pred_term = args.lambda_pred * pred_loss
        msle_term = args.lambda_msle * msle_loss
        rank_term = args.lambda_rank * rank_loss
        tcl_term = args.lambda_tcl * tcl_loss
        orth_term = args.lambda_orth * orth_loss
        loss = pred_term + amp_term + scale_term + msle_term + rank_term + balance_term + tcl_term + orth_term

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_amp_loss += float(amp_metric)
        total_scale_loss += float(scale_metric)
        total_msle_loss += float(msle_loss)
        total_rank_loss += float(rank_loss)
        total_weighted_pred += pred_term.item()
        total_weighted_amp += amp_term.item()
        total_weighted_scale += scale_term.item()
        total_weighted_msle += msle_term.item()
        total_weighted_rank += rank_term.item()
        total_balance_loss += float(lb_loss.detach())
        total_weighted_balance += balance_term.item()
        total_tcl_loss += float(tcl_loss.detach())
        total_orth_loss += float(orth_loss.detach())
        total_weighted_tcl += tcl_term.item()
        total_weighted_orth += orth_term.item()

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(train_loader):
            cmax_idx = int(torch.argmax(cluster_usage).item())
            cmax_val = float(cluster_usage[cmax_idx].item())
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Pred': f'{pred_loss.item():.4f}',
                'Amp': f'{amp_metric.item():.4f}',
                'Scale': f'{scale_metric.item():.4f}',
                'λPred': f'{pred_term.item():.4f}',
                'λAmp': f'{amp_term.item():.4f}',
                'λScale': f'{scale_term.item():.4f}',
                'MSLE': f'{msle_loss.item():.4f}',
                'Rank': f'{rank_loss.item():.4f}',
                'LB': f'{lb_loss.item():.4f}',
                'λLB': f'{balance_term.item():.4f}',
                'TCL': f'{tcl_loss.item():.4f}',
                'Orth': f'{orth_loss.item():.4f}',
                'λTCL': f'{tcl_term.item():.4f}',
                'λOrth': f'{orth_term.item():.4f}',
                'Cmax': f'{cmax_idx}:{cmax_val:.3f}',
            })

    n_batches = len(train_loader)
    avg_usage = None
    if cluster_usage_steps > 0 and cluster_usage_sum is not None:
        avg_usage = (cluster_usage_sum / cluster_usage_steps).cpu().tolist()
    return {
        'loss': total_loss / n_batches,
        'pred_loss': total_pred_loss / n_batches,
        'amp_loss': total_amp_loss / n_batches,
        'scale_loss': total_scale_loss / n_batches,
        'msle_loss': total_msle_loss / n_batches,
        'rank_loss': total_rank_loss / n_batches,
        'weighted_pred': total_weighted_pred / n_batches,
        'weighted_amp': total_weighted_amp / n_batches,
        'weighted_scale': total_weighted_scale / n_batches,
        'weighted_msle': total_weighted_msle / n_batches,
        'weighted_rank': total_weighted_rank / n_batches,
        'load_balance_loss': total_balance_loss / n_batches,
        'weighted_balance': total_weighted_balance / n_batches,
        'tcl_loss': total_tcl_loss / n_batches,
        'orth_loss': total_orth_loss / n_batches,
        'weighted_tcl': total_weighted_tcl / n_batches,
        'weighted_orth': total_weighted_orth / n_batches,
        'cluster_usage': avg_usage
    }


def validate(model, val_loader, criterion, rank_loss_fn, args, data_max, data_min):
    model.eval()
    total_loss = 0.0
    amp_accum = 0.0
    scale_accum = 0.0
    msle_accum = 0.0
    rank_accum = 0.0
    tcl_accum = 0.0
    orth_accum = 0.0
    all_preds = []
    all_targets = []

    log_every = max(1, int(getattr(args, 'log_interval', 10)))
    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc='Validation',
            miniters=log_every,
            mininterval=1.0,
            dynamic_ncols=True,
        ):
            img, text, x, y, coords, time_info = batch

            img = _to_device(img, args.device)
            text = _to_device(text, args.device)
            x = x.to(args.device)
            y = y.to(args.device)
            coords = coords.to(args.device)
            time_info = _to_device(time_info, args.device)

            if x.dim() == 3:
                x = x.unsqueeze(-1)

            pred, details = model(
                X=x,
                I=img,
                T=text,
                coords=coords,
                time_info=time_info,
                return_gates=True
            )

            pred, y = _align_prediction_target(pred, y)

            pred_loss = criterion(pred, y)
            loss = args.lambda_pred * pred_loss

            if args.lambda_amp > 0:
                amp_batch = _amplitude_loss(pred, y)
                loss = loss + args.lambda_amp * amp_batch
                amp_metric = amp_batch
            else:
                amp_metric = _amplitude_loss(pred, y)

            if args.lambda_scale > 0:
                scale_batch = _scale_loss(pred, y, args.scale_epsilon)
                loss = loss + args.lambda_scale * scale_batch
                scale_metric = scale_batch
            else:
                scale_metric = _scale_loss(pred, y, args.scale_epsilon)

            msle_batch = _msle_loss(pred, y)
            rank_batch = _ranking_loss(pred, y, args.rank_margin, rank_loss_fn)
            tcl_batch = _temporal_contrastive_loss(
                            details.get('image_dynamic') if isinstance(details, dict) else None,
                            details.get('poi_dynamic') if isinstance(details, dict) else None,
                            coords,                
                            args.glam_knn_k,        
                            args.contrastive_tau
                        )
            orth_batch = _orthogonality_loss(
                details.get('image_dynamic') if isinstance(details, dict) else None,
                details.get('poi_dynamic') if isinstance(details, dict) else None,
                args.orth_eps
            )
            if tcl_batch is None:
                tcl_batch = _zero_scalar_like(pred_loss)
            if orth_batch is None:
                orth_batch = _zero_scalar_like(pred_loss)

            loss = (
                loss
                + args.lambda_msle * msle_batch
                + args.lambda_rank * rank_batch
                + args.lambda_tcl * tcl_batch
                + args.lambda_orth * orth_batch
            )

            total_loss += loss.item()
            amp_accum += amp_metric.item()
            scale_accum += scale_metric.item()
            msle_accum += msle_batch.item()
            rank_accum += rank_batch.item()
            tcl_accum += tcl_batch.item()
            orth_accum += orth_batch.item()

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(
        all_preds, all_targets,
        denormalize=True,
        data_max=data_max,
        data_min=data_min
    )
    metrics['AMP_MAE'] = torch.mean(
        torch.abs(all_preds.mean(dim=-1) - all_targets.mean(dim=-1))
    ).item()
    metrics['SCALE_LOSS'] = scale_accum / len(val_loader)

    avg_loss = total_loss / len(val_loader)
    avg_amp = amp_accum / len(val_loader)
    avg_scale = scale_accum / len(val_loader)
    avg_msle = msle_accum / len(val_loader)
    avg_rank = rank_accum / len(val_loader)
    avg_tcl = tcl_accum / len(val_loader)
    avg_orth = orth_accum / len(val_loader)
    metrics['MSLE'] = avg_msle
    metrics['RANK_LOSS'] = avg_rank
    metrics['TCL_LOSS'] = avg_tcl
    metrics['ORTH_LOSS'] = avg_orth
    return avg_loss, avg_amp, avg_scale, metrics


def main():
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print('=' * 50)
    print('Training')
    print('=' * 50)
    print(f'Device: {args.device}')
    print(f'Dataset: {args.data}')
    print(f'History/Pred: {args.his_len}/{args.pred_len}')
    print('=' * 50)

    cache_dir = _resolve_default_cache_dir(args.data)
    if not os.path.exists(cache_dir):
        print(f'Cache directory does not exist: {cache_dir}')
        print(f'Please run: python preprocess.py --config {args.config}')
        return
    print(f'Loading cached data from: {cache_dir}')
    try:
        # Use the cache loader from the current cached_dataloader.py implementation.
        train_dataset, val_dataset, test_dataset = load_cached_datasets(
            cache_dir, 
            mask_flow_input=args.mask_flow_input
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    import pickle
    with open(os.path.join(cache_dir, 'base_info.pkl'), 'rb') as f:
        base_info = pickle.load(f)
    data_max = base_info['train_data_max']
    data_min = base_info['train_data_min']

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print('\nCreating model...')
    model = S2F(args).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    print(f'Trainable parameters: {trainable_params / 1e6:.2f}M')

    criterion = nn.MSELoss()
    rank_loss_fn = nn.MarginRankingLoss(margin=0.0)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine_tmax = max(1, int(cosine_epochs * args.cosine_tmax_multiplier))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_tmax,
        eta_min=1e-6
    )

    start_epoch = 0
    best_primary = float('inf')
    best_val_mae = float('inf')
    best_val_amp = float('inf')
    best_val_scale = float('inf')

    if args.resume:
        print(f'\nResuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        best_val_amp = checkpoint.get('best_val_amp', float('inf'))
        best_val_scale = checkpoint.get('best_val_scale', float('inf'))
        best_primary = checkpoint.get('best_primary', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best primary ({args.primary_metric}): {best_primary:.4f}")

    best_model_path = os.path.join(args.save_dir, 'best_model.pth')

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, rank_loss_fn, args, epoch)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(
            f"  Train Loss: {train_metrics['loss']:.4f} "
            f"(Pred: {train_metrics['pred_loss']:.4f}, "
            f"Amp: {train_metrics['amp_loss']:.4f}, "
            f"Scale: {train_metrics['scale_loss']:.4f}, "
            f"MSLE: {train_metrics['msle_loss']:.4f}, "
            f"Rank: {train_metrics['rank_loss']:.4f}, "
            f"LB: {train_metrics['load_balance_loss']:.4f}, "
            f"TCL: {train_metrics['tcl_loss']:.4f}, "
            f"Orth: {train_metrics['orth_loss']:.4f}, "
            f"λPred: {train_metrics['weighted_pred']:.4f}, "
            f"λAmp: {train_metrics['weighted_amp']:.4f}, "
            f"λScale: {train_metrics['weighted_scale']:.4f}, "
            f"λMSLE: {train_metrics['weighted_msle']:.4f}, "
            f"λRank: {train_metrics['weighted_rank']:.4f}, "
            f"λLB: {train_metrics['weighted_balance']:.4f}, "
            f"λTCL: {train_metrics['weighted_tcl']:.4f}, "
            f"λOrth: {train_metrics['weighted_orth']:.4f})"
        )
        if train_metrics.get('cluster_usage'):
            usage_fmt = ', '.join(f"{u:.3f}" for u in train_metrics['cluster_usage'])
            print(f"  Cluster Usage (avg): [{usage_fmt}]")

        if (epoch + 1) % args.val_interval == 0:
            val_loss, avg_amp_loss, avg_scale_loss, val_metrics = validate(
                model, val_loader, criterion, rank_loss_fn, args,
                data_max=data_max,
                data_min=data_min
            )

            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val AMP loss (batch mean): {avg_amp_loss:.4f}')
            print(f'  Val Scale loss (batch mean): {avg_scale_loss:.4f}')
            print(f"  Val MSLE: {val_metrics['MSLE']:.4f}")
            print(f"  Val Rank Loss: {val_metrics['RANK_LOSS']:.4f}")
            print(f"  Val TCL Loss: {val_metrics['TCL_LOSS']:.4f}")
            print(f"  Val Orth Loss: {val_metrics['ORTH_LOSS']:.4f}")
            print(f"  Val MAE: {val_metrics['MAE']:.2f} (real scale)")
            print(f"  Val AMP-MAE: {val_metrics['AMP_MAE']:.2f}")
            print(f"  Val Scale: {val_metrics['SCALE_LOSS']:.2f}")
            if 'Inflow_MAE' in val_metrics and 'Outflow_MAE' in val_metrics:
                print(
                    f"  Val Inflow MAE/RMSE/MAPE: {val_metrics['Inflow_MAE']:.2f} / "
                    f"{val_metrics.get('Inflow_RMSE', float('nan')):.2f} / "
                    f"{val_metrics.get('Inflow_MAPE', float('nan')):.2f}%"
                )
                print(
                    f"  Val Outflow MAE/RMSE/MAPE: {val_metrics['Outflow_MAE']:.2f} / "
                    f"{val_metrics.get('Outflow_RMSE', float('nan')):.2f} / "
                    f"{val_metrics.get('Outflow_MAPE', float('nan')):.2f}%"
                )
            print(f"  Val RMSE: {val_metrics['RMSE']:.2f}")
            if 'RMSLE' in val_metrics:
                print(f"  Val RMSLE: {val_metrics['RMSLE']:.4f}")
            print(f"  Val MAPE: {val_metrics['MAPE']:.2f}%")
            if 'CPC_Node' in val_metrics:
                print(f"  Val CPC-Node: {val_metrics['CPC_Node']:.3f}")
            if 'Spearman' in val_metrics:
                print(f"  Val Spearman: {val_metrics['Spearman']:.3f}")
            print(f"  Val R²: {val_metrics['R2']:.4f}")

            primary_score = _primary_score(args, val_metrics)
            print(f"  Primary ({args.primary_metric}): {primary_score:.4f}")

            required_gap = args.mape_save_delta if args.primary_metric == 'mape' else 0.0
            save_due = primary_score < (best_primary - required_gap)
            if math.isinf(best_primary):
                save_due = True

            if save_due:
                best_primary = primary_score
                best_val_mae = val_metrics['MAE']
                best_val_amp = val_metrics['AMP_MAE']
                best_val_scale = val_metrics['SCALE_LOSS']
                print('New best model saved!')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_primary': best_primary,
                    'best_val_mae': best_val_mae,
                    'best_val_amp': best_val_amp,
                    'best_val_scale': best_val_scale,
                    'val_metrics': val_metrics,
                    'data_max': data_max,
                    'data_min': data_min,
                    'args': vars(args)
                }, best_model_path)
            elif args.primary_metric == 'mape':
                improvement = best_primary - primary_score
                print(
                    f"Current MAPE improved by only {improvement:.4f}, below threshold {args.mape_save_delta:.4f}; not saving"
                )

        _step_scheduler_with_warmup(optimizer, scheduler, args, epoch)

        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_primary': best_primary,
                'best_val_mae': best_val_mae,
                'best_val_amp': best_val_amp,
                'best_val_scale': best_val_scale,
                'data_max': data_max,
                'data_min': data_min,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

    print('\n' + '=' * 50)
    print('Training completed!')
    print(f"Best primary ({args.primary_metric}): {best_primary:.4f}")
    print(f'Best MAE: {best_val_mae:.2f}, Best AMP-MAE: {best_val_amp:.2f}, Best SCALE: {best_val_scale:.4f}')
    print('=' * 50)


if __name__ == '__main__':
    main()