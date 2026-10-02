import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import Dict, List


def _ensure_batch_shape(array: np.ndarray):
    if array.ndim == 2:
        return array[None, ...]
    return array


def _rankdata(array: np.ndarray) -> np.ndarray:
    order = np.argsort(array, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(array), dtype=float)
    values, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        rank_sums = np.bincount(inverse, weights=ranks)
        avg_ranks = rank_sums / counts
        ranks = avg_ranks[inverse]
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) < 1e-8 or np.std(ry) < 1e-8:
        return 0.0
    corr = np.corrcoef(rx, ry)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _nodewise_cpc(pred_np: np.ndarray, target_np: np.ndarray) -> float:
    pred_np = _ensure_batch_shape(pred_np)
    target_np = _ensure_batch_shape(target_np)
    node_min = np.minimum(pred_np, target_np).sum(axis=-1)
    node_sum = pred_np.sum(axis=-1) + target_np.sum(axis=-1)
    cpc_scores = (2.0 * node_min) / (node_sum + 1e-10)
    return float(np.mean(cpc_scores))


def _rmsle(pred_np: np.ndarray, target_np: np.ndarray) -> float:
    pred_clamped = np.clip(pred_np, a_min=0.0, a_max=None)
    target_clamped = np.clip(target_np, a_min=0.0, a_max=None)
    log_pred = np.log1p(pred_clamped)
    log_true = np.log1p(target_clamped)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


def _prepare_denorm_params(data_max, data_min, reference: torch.Tensor):
    """
    Build denormalization tensors compatible with prediction shape.
    """
    max_t = torch.as_tensor(data_max, dtype=reference.dtype, device=reference.device)
    min_t = torch.as_tensor(data_min, dtype=reference.dtype, device=reference.device)

    if max_t.numel() == 1 and min_t.numel() == 1:
        return max_t.reshape(()), min_t.reshape(())

    if max_t.numel() != min_t.numel():
        # Fallback to scalar denormalization when config/cache metadata is inconsistent.
        return max_t.float().mean(), min_t.float().mean()

    if reference.dim() >= 4 and reference.size(-1) == max_t.numel():
        view_shape = [1] * (reference.dim() - 1) + [max_t.numel()]
        return max_t.view(*view_shape), min_t.view(*view_shape)

    # When channel dimension has been reduced (e.g., mean over in/out),
    # apply averaged range to keep scales consistent.
    return max_t.float().mean(), min_t.float().mean()


def _select_channel_scale(scale_value, channel_idx: int):
    scale_t = torch.as_tensor(scale_value)
    if scale_t.numel() == 0:
        return scale_value
    if scale_t.numel() == 1:
        return float(scale_t.reshape(-1)[0].item())
    flat = scale_t.reshape(-1)
    if channel_idx < flat.numel():
        return float(flat[channel_idx].item())
    return float(flat.float().mean().item())


def compute_metrics(pred, target, denormalize=True, data_max=1.0, data_min=0.0):
    """
    Compute MAE, RMSE, MAPE, wMAPE with proper denormalization.
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    pred_raw = pred
    target_raw = target
    
    if pred.dim() == 4:
        if pred.size(-1) == 1:
            pred = pred.squeeze(-1)
        else:
            pred = pred.mean(dim=-1)
    if target.dim() == 4:
        if target.size(-1) == 1:
            target = target.squeeze(-1)
        else:
            target = target.mean(dim=-1)

    pred = torch.clamp(pred, min=0.0)
    target = torch.clamp(target, min=0.0)
    
    if denormalize:
        denorm_max, denorm_min = _prepare_denorm_params(data_max, data_min, pred)
        if not torch.allclose(denorm_max, denorm_min):
            pred = pred * (denorm_max - denorm_min) + denorm_min
            target = target * (denorm_max - denorm_min) + denorm_min
    
    mae = torch.mean(torch.abs(pred - target)).item()
    
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    
    mask = target > 1e-3
    if mask.sum() > 0:
        mape = torch.mean(torch.abs((pred[mask] - target[mask]) / target[mask])) * 100
        mape = mape.item()
    else:
        mape = 0.0

    target_sum = torch.sum(torch.abs(target)).item()
    if target_sum > 1e-3:
        wmape = (torch.sum(torch.abs(pred - target)).item() / target_sum) * 100
    else:
        wmape = 0.0
    
    ss_res = torch.sum((target - pred) ** 2).item()
    ss_tot = torch.sum((target - target.mean()) ** 2).item()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'wMAPE': wmape,
        'R2': r2
    }

    if pred.dim() >= 3:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        metrics['CPC_Node'] = _nodewise_cpc(pred_np, target_np)
        metrics['RMSLE'] = _rmsle(pred_np, target_np)
        vol_true = target_np.mean(axis=(0, -1))
        vol_pred = pred_np.mean(axis=(0, -1))
        metrics['Spearman'] = _spearman_corr(vol_true, vol_pred)

    # Keep channel-specific metrics for two-channel traffic settings (e.g., inflow/outflow).
    if (
        isinstance(pred_raw, torch.Tensor)
        and isinstance(target_raw, torch.Tensor)
        and pred_raw.dim() == 4
        and target_raw.dim() == 4
        and pred_raw.size(-1) == target_raw.size(-1)
        and pred_raw.size(-1) > 1
    ):
        num_channels = pred_raw.size(-1)
        channel_names = ['Inflow', 'Outflow'] + [f'Ch{i}' for i in range(2, num_channels)]
        for ch_idx in range(num_channels):
            ch_pred = pred_raw[..., ch_idx]
            ch_target = target_raw[..., ch_idx]
            ch_max = _select_channel_scale(data_max, ch_idx)
            ch_min = _select_channel_scale(data_min, ch_idx)
            ch_metrics = compute_metrics(
                ch_pred,
                ch_target,
                denormalize=denormalize,
                data_max=ch_max,
                data_min=ch_min,
            )
            ch_prefix = channel_names[ch_idx]
            for key, value in ch_metrics.items():
                metrics[f'{ch_prefix}_{key}'] = value
    
    return metrics



def compute_multihorizon_metrics(pred, target, horizons=[3, 6, 12], denormalize=True, data_max=1.0, data_min=0.0):
    """
    Compute metrics at multiple prediction horizons.
    """
    results = {}
    
    for h in horizons:
        if h <= pred.size(-1):
            h_metrics = compute_metrics(
                pred[:, :, :h],
                target[:, :, :h],
                denormalize=denormalize,
                data_max=data_max,
                data_min=data_min
            )
            results[f'horizon_{h}'] = h_metrics
    
    return results


class MaskedMAELoss(nn.Module):
    """
    Masked MAE loss (ignore zero values).
    """
    def __init__(self, mask_value=0.0):
        super().__init__()
        self.mask_value = mask_value
    
    def forward(self, pred, target):
        mask = (target != self.mask_value).float()
        loss = torch.abs(pred - target) * mask
        return loss.sum() / (mask.sum() + 1e-5)


class MaskedRMSELoss(nn.Module):
    """
    Masked RMSE loss.
    """
    def __init__(self, mask_value=0.0):
        super().__init__()
        self.mask_value = mask_value
    
    def forward(self, pred, target):
        mask = (target != self.mask_value).float()
        loss = ((pred - target) ** 2) * mask
        return torch.sqrt(loss.sum() / (mask.sum() + 1e-5))


def save_checkpoint(state, save_dir, filename='checkpoint.pth', is_best=False):
    """
    Save model checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)
    
    return filepath


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_val_mae': checkpoint.get('best_val_mae', float('inf')),
        'val_metrics': checkpoint.get('val_metrics', {})
    }


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)


class MetricTracker:
    """
    Track and log training metrics.
    """
    def __init__(self, save_dir=None):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rate': []
        }
        self.save_dir = save_dir
    
    def update(self, epoch, train_loss=None, val_loss=None, val_metrics=None, lr=None):
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_metrics is not None:
            self.history['val_metrics'].append(val_metrics)
        if lr is not None:
            self.history['learning_rate'].append(lr)
    
    def save(self, filename='training_history.json'):
        if self.save_dir:
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=4)
    
    def get_best_epoch(self, metric='MAE'):
        if not self.history['val_metrics']:
            return None
        
        scores = [m.get(metric, float('inf')) for m in self.history['val_metrics']]
        return np.argmin(scores)


def get_loss_function(loss_type='mse', **kwargs):
    """
    Get loss function by name.
    """
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.SmoothL1Loss()
    elif loss_type == 'masked_mae':
        return MaskedMAELoss(**kwargs)
    elif loss_type == 'masked_rmse':
        return MaskedRMSELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def denormalize(data, data_max, data_min):
    """
    Denormalize data.
    """
    if data_max == data_min:
        return data + data_min
    return data * (data_max - data_min) + data_min


def normalize(data, data_max, data_min):
    """
    Normalize data.
    """
    if data_max == data_min:
        return data - data_min
    return (data - data_min) / (data_max - data_min)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized