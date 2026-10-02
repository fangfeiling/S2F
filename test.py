import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import time


from models.s2f import S2F
from utils.metrics import compute_metrics
from utils.config_loader import load_config, config_to_args
from utils.cached_dataloader import load_cached_datasets

SPATIAL_KEYS = ('CPC_Node', 'RMSLE', 'Spearman')


def _extract_spatial_view(metrics):
    return {k: metrics[k] for k in SPATIAL_KEYS if k in metrics}


def _align_prediction_target(pred: torch.Tensor, target: torch.Tensor):
    while pred.dim() > target.dim() and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    while target.dim() > pred.dim() and target.size(-1) == 1:
        target = target.squeeze(-1)

    if pred.dim() == 3 and target.dim() == 4:
        target = target.mean(dim=-1)
    elif pred.dim() == 4 and target.dim() == 3:
        pred = pred.mean(dim=-1)
    elif pred.dim() == 4 and target.dim() == 4 and pred.size(-1) != target.size(-1):
        pred = pred.mean(dim=-1)
        target = target.mean(dim=-1)

    return pred, target


def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Scene2Flow Model Testing')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (e.g., config/NYC.conf)')
    
    parser.add_argument('--data', type=str, default='NYC_TAXI', 
                       choices=['BM_TAXI', 'DC_TAXI', 'NYC_TAXI'])
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--no_strict_load', dest='strict_load', action='store_false', default=True,
                       help='Allow non-strict checkpoint loading (for cross-city eval with shape mismatch)')
    
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
                       help='Top-k experts per token for modality MoE (default: 2)')
    parser.add_argument('--feature_dim', type=int, default=32)
    parser.add_argument('--prompt_dim', type=int, default=128,
                       help='Disentangled prompting hidden size')
    parser.add_argument('--image_token_count', type=int, default=8,
                       help='Pseudo image token count for prompt-driven cross attention')
    parser.add_argument('--use_glam', action='store_true', default=True,
                       help='Enable GLAM modulation before MoE')
    parser.add_argument('--glam_knn_k', type=int, default=8,
                       help='Neighborhood size for city-level graph context')
    parser.add_argument('--prototype_tau', type=float, default=0.2,
                       help='Temperature for prototype routing')
    parser.add_argument('--use_prototype_router', action='store_true', default=True,
                       help='Use prototype routing as cluster membership')
    parser.add_argument('--disable_image', action='store_true', default=False,
                       help='Ablation: remove image signal by zeroing image features')
    parser.add_argument('--disable_text', action='store_true', default=False,
                       help='Ablation: remove text signal by zeroing text features')
    parser.add_argument('--use_remoteclip', action='store_true', default=True)
    parser.add_argument('--remoteclip_model', type=str, default='ViT-B-32')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mask_flow_input', action='store_true', default=True,
                       help='Zero historical flows and rely on spatio-temporal priors only')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_predictions', action='store_true', default=True)
    parser.add_argument('--save_modal_predictions', action='store_true', default=False,
                       help='Save per-modality predictions when the model provides them')
    parser.add_argument('--save_routing_details', action='store_true', default=False,
                       help='Save MoE routing (router_probs) and modality_weights for analysis')
    parser.add_argument('--metrics_only', action='store_true', default=False,
                       help='Only save core metrics JSON (disables saving predictions/modal/routing artifacts)')
    parser.add_argument('--routing_topk', type=int, default=32,
                       help='Max top-k highest-error samples to save routing details for')
    parser.add_argument('--routing_random_k', type=int, default=16,
                       help='Random samples (reservoir) to save routing details for')
    parser.add_argument('--routing_selection', type=str, default='topk', choices=['topk', 'random', 'both'],
                       help='Selection strategy for routing detail saving')
    parser.add_argument('--output_dir', type=str, default='test_results/')
    
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
        config_args = config_to_args(config)
        for key, value in vars(config_args).items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    return args


def load_model(args, checkpoint_path):
    """
    Load trained model from checkpoint.
    """
    print(f"\nLoading model from: {checkpoint_path}")
    
    model = S2F(args).to(args.device)
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        try:
            load_ret = model.load_state_dict(state_dict, strict=args.strict_load)
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Model loaded from epoch {epoch} (strict={args.strict_load})")
        except RuntimeError as exc:
            print(f"[WARN] Load failed ({exc}); filtering mismatched tensors and retrying with strict=False.")
            filtered = {}
            model_shapes = {k: v.shape for k, v in model.state_dict().items()}
            for k, v in state_dict.items():
                if k in model_shapes and v.shape != model_shapes[k]:
                    continue
                filtered[k] = v
            load_ret = model.load_state_dict(filtered, strict=False)
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Model loaded from epoch {epoch} (strict=False, filtered mismatched keys)")
            args.strict_load = False
        if not args.strict_load:
            missing = getattr(load_ret, 'missing_keys', [])
            unexpected = getattr(load_ret, 'unexpected_keys', [])
            if missing:
                print(f"  Missing keys (random init): {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {len(unexpected)}")
        
        data_max = checkpoint.get('data_max', None)
        data_min = checkpoint.get('data_min', None)
        
        if data_max is not None and data_min is not None:
            data_min_scalar = data_min.item() if isinstance(data_min, torch.Tensor) else float(data_min)
            data_max_scalar = data_max.item() if isinstance(data_max, torch.Tensor) else float(data_max)
            print(f"Loaded normalization params: min={data_min_scalar:.2f}, max={data_max_scalar:.2f}")
        else:
            print("Warning: No normalization params in checkpoint!")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded (no epoch info)")
        data_max, data_min = None, None
    
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    return model, data_max, data_min


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
    return os.path.join('preprocessed_data', f'{data_name}')


def _maybe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def test_model(model, test_loader, args, data_max, data_min):
    """
    Test model and compute metrics with denormalization, optionally saving routing details.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    modal_pred_buffers = {} if args.save_modal_predictions else None

    total_inference_time = 0.0
    total_samples = 0
    device_is_cuda = isinstance(args.device, str) and ('cuda' in args.device)

    want_details = args.save_modal_predictions or args.save_routing_details
    import heapq, random
    topk_heap = []
    reservoir = []
    sample_counter = 0
    routing_topk = max(0, args.routing_topk)
    routing_rand = max(0, args.routing_random_k)
    selection_mode = args.routing_selection
    routing_summary = {
        'expert_mass': None,
        'total_samples': 0,
        'modality_order': None
    }
    expert_mass_accum = None
    modality_order = None

    print("\nTesting model...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            img, text, x, y, coords, time_info = batch
            
            img = _to_device(img, args.device)
            text = _to_device(text, args.device)
            x = x.to(args.device)
            y = y.to(args.device)
            coords = coords.to(args.device)
            time_info = _to_device(time_info, args.device)
            
            if x.dim() == 3:
                x = x.unsqueeze(-1)
            
            if device_is_cuda:
                torch.cuda.synchronize()
            start_time = time.time()

            if want_details:
                pred, details = model(
                    X=x,
                    I=img,
                    T=text,
                    coords=coords,
                    time_info=time_info,
                    return_gates=True
                )
                modality_preds = details.get('modality_predictions', {})
                if args.save_modal_predictions:
                    for name, tensor in modality_preds.items():
                        modal_pred_buffers.setdefault(name, []).append(tensor.cpu())
                router_probs = details.get('router_probs')
                modality_weights = details.get('modality_weights')
                if modality_order is None:
                    modality_order = details.get('modality_order')
                router_probs_combined = None
                if router_probs is not None:
                    if isinstance(router_probs, dict):
                        stacks = []
                        for v in router_probs.values():
                            stacks.append(v)
                        if stacks:
                            router_probs_combined = torch.stack(stacks, dim=0).mean(dim=0)
                    else:
                        router_probs_combined = router_probs
                if router_probs_combined is not None:
                    if expert_mass_accum is None:
                        expert_mass_accum = torch.zeros(
                            router_probs_combined.shape[-1],
                            device=router_probs_combined.device
                        )
                    expert_mass_accum += router_probs_combined.sum(dim=(0, 1, 2))
                routing_summary['total_samples'] += x.size(0)
            else:
                pred = model(
                    X=x,
                    I=img,
                    T=text,
                    coords=coords,
                    time_info=time_info
                )
                router_probs = None
                modality_weights = None

            if device_is_cuda:
                torch.cuda.synchronize()
            total_inference_time += (time.time() - start_time)
            total_samples += x.size(0)
            
            pred, y = _align_prediction_target(pred, y)
            
            abs_err = (pred - y).abs()
            reduce_dims = tuple(range(1, abs_err.dim()))
            sample_scores = abs_err.mean(dim=reduce_dims)

            all_predictions.append(pred.cpu())
            all_targets.append(y.cpu())

            if args.save_routing_details and (router_probs is not None or modality_weights is not None):
                for i in range(pred.size(0)):
                    score = sample_scores[i].detach().item()
                    rp_to_save = None
                    if router_probs_combined is not None:
                        rp_to_save = router_probs_combined[i].detach().cpu()
                    record = {
                        'sample_id': sample_counter,
                        'score': score,
                        'pred': pred[i].detach().cpu(),
                        'target': y[i].detach().cpu(),
                        'coords': coords[i].detach().cpu(),
                        'router_probs': rp_to_save,
                        'modality_weights': modality_weights[i].detach().cpu() if modality_weights is not None else None,
                    }
                    if selection_mode in ('topk', 'both'):
                        if routing_topk > 0:
                            if len(topk_heap) < routing_topk:
                                heapq.heappush(topk_heap, (score, sample_counter, record))
                            else:
                                if score > topk_heap[0][0]:
                                    heapq.heapreplace(topk_heap, (score, sample_counter, record))
                    if selection_mode in ('random', 'both'):
                        if routing_rand > 0:
                            if len(reservoir) < routing_rand:
                                reservoir.append(record)
                            else:
                                j = random.randint(0, sample_counter)
                                if j < routing_rand:
                                    reservoir[j] = record
                    sample_counter += 1
            else:
                sample_counter += pred.size(0)

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    modal_predictions = None
    if modal_pred_buffers:
        modal_predictions = {
            name: torch.cat(buf, dim=0).numpy()
            for name, buf in modal_pred_buffers.items()
        }
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    if data_max is not None and data_min is not None:
        metrics = compute_metrics(
            predictions, targets,
            denormalize=True,
            data_max=data_max,
            data_min=data_min
        )
        print("Metrics computed on REAL SCALE (denormalized)")
    else:
        metrics = compute_metrics(
            predictions, targets,
            denormalize=False
        )
        print("Warning: Metrics computed on normalized scale!")
    
    routing_summary['modality_order'] = modality_order
    if expert_mass_accum is not None:
        routing_summary['expert_mass'] = expert_mass_accum.detach().cpu().numpy().tolist()

    selected = []
    if args.save_routing_details:
        if selection_mode in ('topk', 'both'):
            selected.extend([item for _, _, item in sorted(topk_heap, key=lambda t: -t[0])])
        if selection_mode in ('random', 'both'):
            selected.extend(reservoir)

    if total_samples > 0:
        avg_time_ms = (total_inference_time / total_samples) * 1000.0
        metrics['Inference_Time_ms'] = avg_time_ms
        print(f"Average Inference Time: {avg_time_ms:.4f} ms/sample")

    return predictions.numpy(), targets.numpy(), metrics, modal_predictions, routing_summary, selected



def save_results(predictions, targets, metrics, args, modal_predictions=None, routing_summary=None, selected_samples=None):
    """
    Save test results and optional routing details.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_predictions:
        np.save(os.path.join(args.output_dir, 'predictions.npy'), predictions)
        np.save(os.path.join(args.output_dir, 'targets.npy'), targets)
        print(f"\nPredictions saved to {args.output_dir}")

    if modal_predictions:
        modal_path = os.path.join(args.output_dir, 'modal_predictions.npz')
        np.savez(modal_path, **modal_predictions)
        print(f"Per-modality predictions saved to {modal_path}")

    if routing_summary is not None and (routing_summary.get('expert_mass') is not None):
        summary_file = os.path.join(args.output_dir, 'routing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(routing_summary, f, indent=2)
        print(f"Routing summary saved to {summary_file}")

    if selected_samples:
        sel_dir = os.path.join(args.output_dir, 'routing_samples')
        os.makedirs(sel_dir, exist_ok=True)
        index_entries = []
        for rec in selected_samples:
            sid = rec['sample_id']
            tag = f"sample_{sid}"
            path = os.path.join(sel_dir, f"{tag}.npz")
            np.savez(
                path,
                pred=rec['pred'].numpy(),
                target=rec['target'].numpy(),
                coords=rec['coords'].numpy(),
                router_probs=None if rec.get('router_probs') is None else rec['router_probs'].numpy(),
                modality_weights=None if rec.get('modality_weights') is None else rec['modality_weights'].numpy(),
                score=_maybe_float(rec.get('score'))
            )
            index_entries.append({
                'sample_id': sid,
                'file': os.path.relpath(path, args.output_dir),
                'score': _maybe_float(rec.get('score'))
            })
        index_file = os.path.join(args.output_dir, 'routing_samples', 'index.json')
        with open(index_file, 'w') as f:
            json.dump(index_entries, f, indent=2)
        print(f"Selected routing samples saved to {sel_dir}")

    metrics_file = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

    spatial_metrics = _extract_spatial_view(metrics)
    if spatial_metrics and (not getattr(args, 'metrics_only', False)):
        spatial_file = os.path.join(args.output_dir, 'spatial_metrics.json')
        with open(spatial_file, 'w') as f:
            json.dump(spatial_metrics, f, indent=4)
        print(f"Spatial metrics saved to {spatial_file}")
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    if spatial_metrics:
        print("-" * 24)
        for key, value in spatial_metrics.items():
            print(f"{key}: {value:.4f}")
    print("="*60)


def main():
    args = get_args()

    if getattr(args, 'metrics_only', False):
        args.save_predictions = False
        args.save_modal_predictions = False
        args.save_routing_details = False
    
    print("="*60)
    print("E2-CSTP Model Testing")
    print("="*60)
    print(f"Dataset: {args.data}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("="*60)
    
    print("\nLoading test dataset...")
    cache_dir = _resolve_default_cache_dir(args.data)
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        print(f"Please run: python preprocess.py --config {args.config}")
        return
    _, _, test_dataset = load_cached_datasets(
        cache_dir,
        mask_flow_input=args.mask_flow_input,
    )
    data_max = test_dataset.train_data_max
    data_min = test_dataset.train_data_min
    print(f"Cached test dataset loaded from {cache_dir}")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    model, ckpt_data_max, ckpt_data_min = load_model(args, args.checkpoint)
    data_max = ckpt_data_max or data_max
    data_min = ckpt_data_min or data_min
    
    start_time = time.time()
    predictions, targets, metrics, modal_predictions, routing_summary, selected_samples = test_model(model, test_loader, args, data_max, data_min)
    test_time = time.time() - start_time
    
    print(f"\nTesting completed in {test_time:.2f}s")
    print(f"Average time per sample: {test_time/len(test_dataset)*1000:.2f}ms")
    
    save_results(
        predictions,
        targets,
        metrics,
        args,
        modal_predictions=modal_predictions,
        routing_summary=routing_summary if args.save_routing_details else None,
        selected_samples=selected_samples if args.save_routing_details else None
    )
    
    print("\n" + "="*60)
    print("Testing completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()