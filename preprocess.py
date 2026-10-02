import torch
import argparse
import os
import pickle
import json
from tqdm import tqdm
from utils.dataloader import TaxiFlowDataset, TaxiFlowTrainDataset, TaxiFlowValDataset, TaxiFlowTestDataset
from utils.config_loader import load_config, config_to_args


def _resolve_cache_dir(data_name: str) -> str:
    return f"preprocessed_data/{data_name}"


def _precompute_modal_embeddings(cache_dir, args, base_dataset, train_dataset):
    from layers.encoders import FeatureExtractionModule

    device_str = getattr(args, 'embedding_device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        print(f"[WARN] Requested device {device_str} is unavailable; falling back to cpu")
        device_str = 'cpu'
    device = torch.device(device_str)

    batch_size = max(1, int(getattr(args, 'embedding_batch_size', 64)))
    print("\n[6/6] Precomputing multimodal embedding caches...")
    print(f"  Device: {device} | Image encoding batch size: {batch_size}")

    extractor = FeatureExtractionModule(
        use_remoteclip=getattr(args, 'use_remoteclip', True),
        remoteclip_model=getattr(args, 'remoteclip_model', 'ViT-B-32'),
        image_token_count=getattr(args, 'image_token_count', 8),
        allow_offline_fallback=True,
    ).to(device)
    extractor.eval()
    extractor.freeze_pretrained_backbones()

    image_keys = list(base_dataset.image_cache.keys())
    image_backbone_feature_cache = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(image_keys), batch_size), desc='Encoding image cache'):
            batch_keys = image_keys[start:start + batch_size]
            batch_imgs = torch.stack([base_dataset.image_cache[k] for k in batch_keys], dim=0).to(device)
            batch_imgs = batch_imgs.unsqueeze(0)
            # Cache raw RemoteCLIP outputs; projection layers remain trainable in main model.
            img_feat = extractor.img_encoder(batch_imgs)
            img_feat = img_feat.squeeze(0).detach().cpu().to(torch.float16)
            for i, k in enumerate(batch_keys):
                image_backbone_feature_cache[k] = img_feat[i]

        text_ids = train_dataset.text_data.to(device)
        text_attention_mask = (text_ids != 0).to(torch.long)
        # Cache raw BERT token embeddings; text_proj stays trainable in main model.
        text_embeddings, text_mask = extractor.text_encoder(
            text_ids,
            attention_mask=text_attention_mask,
            return_all_tokens=True,
        )
        text_embeddings = text_embeddings.squeeze(0).detach().cpu().to(torch.float16)
        text_mask = text_mask.squeeze(0).detach().cpu().to(torch.bool)

    image_emb_path = os.path.join(cache_dir, 'image_embedding_cache.pt')
    text_emb_path = os.path.join(cache_dir, 'text_embedding_cache.pt')
    torch.save(
        {
            'image_backbone_features': image_backbone_feature_cache,
            'dtype': 'float16',
            'projection_applied': False,
        },
        image_emb_path,
    )
    torch.save(
        {
            'text_backbone_embeddings': text_embeddings,
            'text_mask': text_mask,
            'dtype': 'float16',
            'projection_applied': False,
        },
        text_emb_path,
    )

    meta = {
        'image_keys': len(image_backbone_feature_cache),
        'text_shape': list(text_embeddings.shape),
        'image_dim': int(next(iter(image_backbone_feature_cache.values())).numel()) if image_backbone_feature_cache else 0,
        'token_count': int(getattr(args, 'image_token_count', 8)),
        'projection_applied_in_cache': False,
    }
    with open(os.path.join(cache_dir, 'embedding_cache_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved image embedding cache: {image_emb_path}")
    print(f"  Saved text embedding cache: {text_emb_path}")
    print(f"  Precompute summary: {meta}")

def preprocess_dataset(args):
    print("=" * 60)
    print("Starting dataset preprocessing")
    print(f"Dataset: {args.data} | Nodes: {args.num_nodes}")
    print("=" * 60)

    cache_dir = _resolve_cache_dir(args.data)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    train_ratio = getattr(args, 'train_ratio', 0.6)
    val_ratio = getattr(args, 'val_ratio', 0.2)

    print("\n[1/5] Loading base dataset with temporal image nearest-neighbor matching...")
    base_dataset = TaxiFlowDataset(
        taxi_flow_path=args.taxi_flow_path,
        geo_path=args.geo_path,
        text_dir=args.text_dir,
        image_dir=args.image_dir,
        his_len=args.his_len,
        pred_len=args.pred_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    base_info = {
        'num_nodes': base_dataset.num_nodes,
        'num_timesteps': base_dataset.num_timesteps,
        'nums': base_dataset.nums,
        'train_data_max': base_dataset.train_data_max,
        'train_data_min': base_dataset.train_data_min,
        'coords': base_dataset.coords,
        'region_ids': base_dataset.region_ids,
        'text_format': getattr(base_dataset, 'text_format', 'unknown'),
        'text_max_len': getattr(base_dataset, 'text_max_len', None),
    }

    print("\n[2/5] Saving global deduplicated image cache...")
    cache_file_img = os.path.join(cache_dir, 'image_cache.pkl')
    with open(cache_file_img, 'wb') as f:
        pickle.dump(base_dataset.image_cache, f)
    print(f"  Image cache saved, size: {os.path.getsize(cache_file_img) / 1024 / 1024:.2f} MB")

    print("\n[3/5] Preprocessing training split...")
    train_dataset = TaxiFlowTrainDataset(base_dataset)
    train_data = {
        'img_keys': train_dataset.img_keys,  # Replaces the previous raw img_data payload.
        'text_data': train_dataset.text_data,
        'x_samples_data': train_dataset.x_samples_data,
        'y_samples_data': train_dataset.y_samples_data,
        'coords_data': train_dataset.coords_data,
        'time_info_data': train_dataset.time_info_data,
        'train_data_max': train_dataset.train_data_max,
        'train_data_min': train_dataset.train_data_min,
    }
    with open(os.path.join(cache_dir, 'train_dataset.pkl'), 'wb') as f:
        pickle.dump(train_data, f)

    print("\n[4/5] Preprocessing validation split...")
    val_dataset = TaxiFlowValDataset(base_dataset)
    val_data = {
        'img_keys': val_dataset.img_keys,
        'text_data': val_dataset.text_data,
        'x_samples_data': val_dataset.x_samples_data,
        'y_samples_data': val_dataset.y_samples_data,
        'coords_data': val_dataset.coords_data,
        'time_info_data': val_dataset.time_info_data,
        'train_data_max': val_dataset.train_data_max,
        'train_data_min': val_dataset.train_data_min,
    }
    with open(os.path.join(cache_dir, 'val_dataset.pkl'), 'wb') as f:
        pickle.dump(val_data, f)

    print("\n[5/5] Preprocessing test split...")
    test_dataset = TaxiFlowTestDataset(base_dataset)
    test_data = {
        'img_keys': test_dataset.img_keys,
        'text_data': test_dataset.text_data,
        'x_samples_data': test_dataset.x_samples_data,
        'y_samples_data': test_dataset.y_samples_data,
        'coords_data': test_dataset.coords_data,
        'time_info_data': test_dataset.time_info_data,
        'train_data_max': test_dataset.train_data_max,
        'train_data_min': test_dataset.train_data_min,
    }
    with open(os.path.join(cache_dir, 'test_dataset.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    with open(os.path.join(cache_dir, 'base_info.pkl'), 'wb') as f:
        pickle.dump(base_info, f)

    _precompute_modal_embeddings(cache_dir, args, base_dataset, train_dataset)

    print("\n" + "=" * 60)
    print("Dataset preprocessing completed. The artifacts are ready for v2 training/testing.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--embedding_batch_size', type=int, default=64,
                        help='Batch size for image embedding precomputation')
    parser.add_argument('--embedding_device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device used for embedding precomputation')
    cli_args = parser.parse_args()

    args = config_to_args(load_config(cli_args.config))
    args.embedding_batch_size = cli_args.embedding_batch_size
    args.embedding_device = cli_args.embedding_device
    preprocess_dataset(args)

if __name__ == '__main__':
    main()