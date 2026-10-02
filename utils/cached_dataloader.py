import torch
from torch.utils.data import Dataset
import pickle
import os


def _safe_torch_load(path, map_location='cpu'):
    """Prefer safe loading mode when supported by current PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location=map_location)


class CachedDataset(Dataset):
    def __init__(
        self,
        cache_path,
        split='train',
        mask_flow_input=False,
        image_embedding_cache=None,
        text_embedding_cache=None,
    ):
        self.cache_path = cache_path
        self.split = split
        self.mask_flow_input = mask_flow_input
        self.image_embedding_cache = image_embedding_cache
        self.text_embedding_cache = text_embedding_cache

        if self.image_embedding_cache is None or self.text_embedding_cache is None:
            raise ValueError('v2 mode requires precomputed image/text embedding caches')

        cache_file = os.path.join(cache_path, f'{split}_dataset.pkl')
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        print(f"Loading cached {split} dataset: {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        # Restored layout stores lightweight img_keys instead of raw img_data.
        self.img_keys = data['img_keys']
        self.text_data = data['text_data']
        self.x_samples_data = data['x_samples_data']
        self.y_samples_data = data['y_samples_data']
        self.coords_data = data['coords_data']
        self.time_info_data = data['time_info_data']
        self.train_data_max = data['train_data_max']
        self.train_data_min = data['train_data_min']

        if self.mask_flow_input:
            self.x_samples_data = torch.zeros_like(self.x_samples_data)

    def __len__(self):
        return self.x_samples_data.shape[0]

    def __getitem__(self, idx):
        flow = self.x_samples_data[idx]
        if self.mask_flow_input:
            flow = torch.zeros_like(flow)

        keys = self.img_keys[idx]

        img_backbone_bank = self.image_embedding_cache['image_backbone_features']
        dynamic_img_data = {
            'backbone_features': torch.stack([img_backbone_bank[k] for k in keys], dim=0),
        }
        text_payload = {
            'backbone_embeddings': self.text_embedding_cache['text_backbone_embeddings'],
            'mask': self.text_embedding_cache['text_mask'],
        }

        return (
            dynamic_img_data,  # (N, C, H, W)
            text_payload,
            flow,  # (N, his_len, 1)
            self.y_samples_data[idx],  # (N, pred_len, 1)
            self.coords_data,  # (N, 2)
            {k: v[idx] for k, v in self.time_info_data.items()}
        )


def load_cached_datasets(cache_dir, mask_flow_input: bool = False):
    image_embedding_cache = None
    text_embedding_cache = None

    image_emb_path = os.path.join(cache_dir, 'image_embedding_cache.pt')
    text_emb_path = os.path.join(cache_dir, 'text_embedding_cache.pt')
    if not os.path.exists(image_emb_path) or not os.path.exists(text_emb_path):
        raise FileNotFoundError(
            f"V2 mode requires precomputed embedding caches. Run preprocess.py first.\n"
            f"Missing file: {image_emb_path if not os.path.exists(image_emb_path) else text_emb_path}"
        )
    print("Loading precomputed image/text embedding caches...")
    image_embedding_cache = _safe_torch_load(image_emb_path, map_location='cpu')
    text_embedding_cache = _safe_torch_load(text_emb_path, map_location='cpu')

    train_dataset = CachedDataset(
        cache_dir,
        'train',
        mask_flow_input,
        image_embedding_cache=image_embedding_cache,
        text_embedding_cache=text_embedding_cache,
    )
    val_dataset = CachedDataset(
        cache_dir,
        'val',
        mask_flow_input,
        image_embedding_cache=image_embedding_cache,
        text_embedding_cache=text_embedding_cache,
    )
    test_dataset = CachedDataset(
        cache_dir,
        'test',
        mask_flow_input,
        image_embedding_cache=image_embedding_cache,
        text_embedding_cache=text_embedding_cache,
    )

    return train_dataset, val_dataset, test_dataset