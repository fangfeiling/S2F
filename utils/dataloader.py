import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
import glob
from torchvision import transforms
from typing import Dict, List, Tuple

try:
    from transformers import BertTokenizer
except Exception:
    BertTokenizer = None


class TaxiFlowDataset(Dataset):
    """
    Base dataset for multi-modal taxi flow data.
    Handles flow, coordinates, POI text, and remote sensing images.
    """

    def __init__(self,
                 taxi_flow_path: str,
                 geo_path: str,
                 text_dir: str,
                 image_dir: str,
                 his_len: int = 12,
                 pred_len: int = 12,
                 transform=None,
                 mask_flow_input: bool = True,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 text_max_len: int = 200,
                 bert_model_path: str = './checkpoints/bert-base-uncased'):

        self.taxi_flow_path = taxi_flow_path
        self.geo_path = geo_path
        self.text_dir = text_dir
        self.image_dir = image_dir
        self.his_len = his_len
        self.pred_len = pred_len
        self.mask_flow_input = mask_flow_input
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.text_max_len = int(text_max_len)
        self.bert_model_path = bert_model_path

        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self._load_data()

        print(f"Dataset loaded: {self.num_nodes} nodes, {self.num_timesteps} timesteps, feature_dim: {self.flow_dim}")
        print(f"Total samples per node: {self.nums}")
        print(f"Coordinate mean: {self.coords.mean(axis=0)}")

    def _load_data(self):
        # 1. Load flows with automatic support for flow.npy and timestamps.csv.
        taxi_dir = os.path.dirname(self.taxi_flow_path) if self.taxi_flow_path.endswith('.csv') else self.taxi_flow_path
        flow_npy_path = os.path.join(taxi_dir, 'flow.npy')
        time_csv_path = os.path.join(taxi_dir, 'timestamps.csv')

        if not os.path.exists(flow_npy_path) or not os.path.exists(time_csv_path):
            raise FileNotFoundError(f"Missing file: {flow_npy_path} or {time_csv_path}")

        self.flow_data = np.load(flow_npy_path).astype(np.float32)
        df_time = pd.read_csv(time_csv_path)
        self.timestamps = pd.to_datetime(df_time['timestamp'])

        # 2. Load GEO metadata and align region ids.
        if os.path.exists(self.geo_path):
            df_geo = pd.read_csv(self.geo_path)
            self.region_ids = [int(rid) for rid in df_geo['region_id']]
            geo_dict = {int(row['region_id']): (row['center_lon'], row['center_lat'])
                        for _, row in df_geo.iterrows()}
            self.coords = np.array([geo_dict.get(rid, (0.0, 0.0)) for rid in self.region_ids], dtype=np.float32)
        else:
            raise FileNotFoundError(f"Missing geo file required for node alignment: {self.geo_path}")

        # 3. Normalize shapes and apply feature-wise scaling.
        if len(self.flow_data.shape) == 2:
            self.num_timesteps, self.num_nodes = self.flow_data.shape
            self.flow_dim = 1
            self.flow_data = np.expand_dims(self.flow_data, axis=-1)
        else:
            self.num_timesteps, self.num_nodes, self.flow_dim = self.flow_data.shape

        self.nums = self.num_timesteps - self.his_len - self.pred_len

        train_end = int(self.nums * self.train_ratio)
        train_data_part = self.flow_data[:train_end + self.his_len]

        # Feature-wise normalization using min/max over the feature axis.
        self.train_data_min = np.min(train_data_part, axis=(0, 1)).astype(np.float32)
        self.train_data_max = np.max(train_data_part, axis=(0, 1)).astype(np.float32)

        denominator = self.train_data_max - self.train_data_min
        denominator = np.where(denominator == 0, 1.0, denominator)
        self.normalized_data = (self.flow_data - self.train_data_min) / denominator

        # 4. Build deduplicated image cache and time-alignment index.
        self.image_cache = {}
        self.time_to_img_key = []

        self.text_token_cache = {}
        self.text_format = 'char_ord'
        self._init_text_token_cache()

        print("Building remote-sensing image time-alignment cache...")
        for i, rid in enumerate(self.region_ids):
            img_paths = glob.glob(os.path.join(self.image_dir, f"S2_{rid}_*.png"))
            available_dates = []
            path_dict = {}
            for p in img_paths:
                basename = os.path.basename(p)
                try:
                    date_str = basename.split('_')[-1].split('.')[0]
                    date_obj = pd.to_datetime(date_str, format='%Y%m%d')
                    available_dates.append(date_obj)
                    path_dict[date_obj] = basename
                except:
                    continue

            available_dates.sort()

            if not available_dates:
                dummy_key = f"dummy_{rid}"
                if dummy_key not in self.image_cache:
                    self.image_cache[dummy_key] = self.transform(np.zeros((224, 224, 3), dtype=np.uint8))
                self.time_to_img_key.append([dummy_key] * self.nums)
                continue

            for d in available_dates:
                basename = path_dict[d]
                if basename not in self.image_cache:
                    img = cv2.imread(os.path.join(self.image_dir, basename), cv2.IMREAD_COLOR)
                    if img is None:
                        img = np.zeros((224, 224, 3), dtype=np.uint8)
                    else:
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.image_cache[basename] = self.transform(img)

            node_keys = []
            for start_idx in range(self.nums):
                current_time = pd.to_datetime(self.timestamps.iloc[start_idx + self.his_len - 1])
                diffs = [abs((d - current_time).total_seconds()) for d in available_dates]
                closest_date = available_dates[np.argmin(diffs)]
                node_keys.append(path_dict[closest_date])

            self.time_to_img_key.append(node_keys)

    def _read_region_text(self, region_id: int) -> str:
        text_name = f"poi_text_{region_id}.txt"
        text_path = os.path.join(self.text_dir, text_name)
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                return f.read()
        return "No POI data available for this region."

    def _init_text_token_cache(self):
        tokenizer = None
        if BertTokenizer is not None:
            try:
                tokenizer = BertTokenizer.from_pretrained(
                    self.bert_model_path,
                    local_files_only=True
                )
                self.text_format = 'bert_token_ids'
                print(f"Building text token cache with local BERT tokenizer: {self.bert_model_path}")
            except Exception:
                tokenizer = None

        if tokenizer is None:
            self.text_format = 'char_ord'
            print("[WARN] No usable BERT tokenizer found; falling back to character ord token cache.")

        for rid in self.region_ids:
            text_data = self._read_region_text(rid)
            if tokenizer is not None:
                encoded = tokenizer(
                    text_data,
                    truncation=True,
                    padding='max_length',
                    max_length=self.text_max_len,
                    return_tensors='pt'
                )
                token_ids = encoded['input_ids'][0].to(torch.long)
                self.text_token_cache[rid] = token_ids.unsqueeze(0)
            else:
                text_bytes = [ord(c) for c in text_data]
                text_bytes = text_bytes[:self.text_max_len] if len(text_bytes) >= self.text_max_len else text_bytes + [0] * (self.text_max_len - len(text_bytes))
                self.text_token_cache[rid] = torch.tensor(text_bytes, dtype=torch.long).unsqueeze(0)

    def __len__(self):
        return len(self.region_ids)

    def __getitem__(self, idx):
        region_id = self.region_ids[idx]
        img_tensor = None

        text_tensor = self.text_token_cache[region_id]

        time_series = self.normalized_data[:, idx, :]  # Extract multivariate temporal features.
        x_samples, y_samples, time_features = [], [], []

        for start_idx in range(self.nums):
            x_samples.append(time_series[start_idx: start_idx + self.his_len])
            y_samples.append(time_series[start_idx + self.his_len: start_idx + self.his_len + self.pred_len])

            window_timestamps = self.timestamps[start_idx: start_idx + self.his_len]
            pred_timestamps = self.timestamps[
                start_idx + self.his_len: start_idx + self.his_len + self.pred_len
            ]
            hist_accessor = window_timestamps.dt if isinstance(window_timestamps, pd.Series) else window_timestamps
            pred_accessor = pred_timestamps.dt if isinstance(pred_timestamps, pd.Series) else pred_timestamps
            time_feat = {
                'month': getattr(hist_accessor, 'month').values.astype(np.float32),
                'day': getattr(hist_accessor, 'day').values.astype(np.float32),
                'hour': getattr(hist_accessor, 'hour').values.astype(np.float32),
                'minute': getattr(hist_accessor, 'minute').values.astype(np.float32),
                'weekday': getattr(hist_accessor, 'weekday').values.astype(np.float32),
                'pred_month': getattr(pred_accessor, 'month').values.astype(np.float32),
                'pred_day': getattr(pred_accessor, 'day').values.astype(np.float32),
                'pred_hour': getattr(pred_accessor, 'hour').values.astype(np.float32),
                'pred_minute': getattr(pred_accessor, 'minute').values.astype(np.float32),
                'pred_weekday': getattr(pred_accessor, 'weekday').values.astype(np.float32),
            }
            time_features.append(time_feat)

        # No extra np.expand_dims is needed because .npy data already includes the feature axis.
        x_samples_tensor = torch.tensor(np.array(x_samples), dtype=torch.float32)
        if self.mask_flow_input:
            x_samples_tensor = torch.zeros_like(x_samples_tensor)
        y_samples_tensor = torch.tensor(np.array(y_samples), dtype=torch.float32)

        coords = torch.tensor(self.coords[idx], dtype=torch.float32)
        time_info = {
            key: torch.stack([torch.tensor(tf[key], dtype=torch.float32) for tf in time_features])
            for key in time_features[0].keys()
        }

        return img_tensor, text_tensor, x_samples_tensor, y_samples_tensor, coords, time_info


class TaxiFlowTrainDataset(Dataset):
    def __init__(self, original_dataset, mask_flow_input=None):
        self.original_dataset = original_dataset
        self.num_nodes = len(original_dataset)
        self.train_num_time_steps = int(original_dataset.nums * getattr(original_dataset, 'train_ratio', 0.8))
        self.train_data_max = original_dataset.train_data_max
        self.train_data_min = original_dataset.train_data_min
        if mask_flow_input is None:
            self.mask_flow_input = getattr(original_dataset, 'mask_flow_input', False)
        else:
            self.mask_flow_input = bool(mask_flow_input)
        self._reorganize_data()

    def _reorganize_data(self):
        text_list, x_list, y_list, coords_list, time_info_list = [], [], [], [], []
        from tqdm import tqdm
        for i in tqdm(range(self.num_nodes), desc="Loading train nodes"):
            _, text, x, y, coords, time_info = self.original_dataset[i]
            text_list.append(text[:1])
            x_list.append(x[:self.train_num_time_steps])
            y_list.append(y[:self.train_num_time_steps])
            coords_list.append(coords)
            time_info_list.append({k: v[:self.train_num_time_steps] for k, v in time_info.items()})

        self.text_data = torch.stack(text_list, dim=1)
        self.x_samples_data = torch.stack(x_list, dim=1)
        if self.mask_flow_input:
            self.x_samples_data = torch.zeros_like(self.x_samples_data)
        self.y_samples_data = torch.stack(y_list, dim=1)
        self.coords_data = torch.stack(coords_list, dim=0)
        self.time_info_data = {k: torch.stack([time_info_list[i][k] for i in range(self.num_nodes)], dim=1) for k in
                               time_info_list[0].keys()}

        self.img_keys = []
        for t in range(self.train_num_time_steps):
            self.img_keys.append([self.original_dataset.time_to_img_key[n][t] for n in range(self.num_nodes)])

    def __len__(self):
        return self.train_num_time_steps

    def __getitem__(self, idx):
        keys = self.img_keys[idx]
        cache = getattr(self, 'image_cache', self.original_dataset.image_cache)
        dynamic_img_data = torch.stack([cache[k] for k in keys], dim=0)

        return (
            dynamic_img_data,
            self.text_data[0],
            self.x_samples_data[idx],
            self.y_samples_data[idx],
            self.coords_data,
            {k: v[idx] for k, v in self.time_info_data.items()}
        )


class TaxiFlowValDataset(Dataset):
    def __init__(self, original_dataset, mask_flow_input=None):
        self.original_dataset = original_dataset
        self.num_nodes = len(original_dataset)
        self.train_num_time_steps = int(original_dataset.nums * getattr(original_dataset, 'train_ratio', 0.8))
        self.val_num_time_steps = int(original_dataset.nums * getattr(original_dataset, 'val_ratio', 0.1))
        self.train_data_max = original_dataset.train_data_max
        self.train_data_min = original_dataset.train_data_min
        if mask_flow_input is None:
            self.mask_flow_input = getattr(original_dataset, 'mask_flow_input', False)
        else:
            self.mask_flow_input = bool(mask_flow_input)
        self._reorganize_data()

    def _reorganize_data(self):
        text_list, x_list, y_list, coords_list, time_info_list = [], [], [], [], []
        from tqdm import tqdm
        for i in tqdm(range(self.num_nodes), desc="Loading val nodes"):
            _, text, x, y, coords, time_info = self.original_dataset[i]
            text_list.append(text[:1])
            x_list.append(x[self.train_num_time_steps: self.train_num_time_steps + self.val_num_time_steps])
            y_list.append(y[self.train_num_time_steps: self.train_num_time_steps + self.val_num_time_steps])
            coords_list.append(coords)
            time_info_list.append(
                {k: v[self.train_num_time_steps: self.train_num_time_steps + self.val_num_time_steps] for k, v in
                 time_info.items()})

        self.text_data = torch.stack(text_list, dim=1)
        self.x_samples_data = torch.stack(x_list, dim=1)
        if self.mask_flow_input:
            self.x_samples_data = torch.zeros_like(self.x_samples_data)
        self.y_samples_data = torch.stack(y_list, dim=1)
        self.coords_data = torch.stack(coords_list, dim=0)
        self.time_info_data = {k: torch.stack([time_info_list[i][k] for i in range(self.num_nodes)], dim=1) for k in
                               time_info_list[0].keys()}

        self.img_keys = []
        for t in range(self.val_num_time_steps):
            global_idx = self.train_num_time_steps + t
            self.img_keys.append([self.original_dataset.time_to_img_key[n][global_idx] for n in range(self.num_nodes)])

    def __len__(self):
        return self.val_num_time_steps

    def __getitem__(self, idx):
        keys = self.img_keys[idx]
        cache = getattr(self, 'image_cache', self.original_dataset.image_cache)
        dynamic_img_data = torch.stack([cache[k] for k in keys], dim=0)
        return (dynamic_img_data, self.text_data[0], self.x_samples_data[idx], self.y_samples_data[idx],
                self.coords_data, {k: v[idx] for k, v in self.time_info_data.items()})


class TaxiFlowTestDataset(Dataset):
    def __init__(self, original_dataset, mask_flow_input=None):
        self.original_dataset = original_dataset
        self.num_nodes = len(original_dataset)
        self.train_num_time_steps = int(original_dataset.nums * getattr(original_dataset, 'train_ratio', 0.8))
        self.val_num_time_steps = int(original_dataset.nums * getattr(original_dataset, 'val_ratio', 0.1))
        self.test_num_time_steps = original_dataset.nums - self.train_num_time_steps - self.val_num_time_steps
        self.train_data_max = original_dataset.train_data_max
        self.train_data_min = original_dataset.train_data_min
        if mask_flow_input is None:
            self.mask_flow_input = getattr(original_dataset, 'mask_flow_input', False)
        else:
            self.mask_flow_input = bool(mask_flow_input)
        self._reorganize_data()

    def _reorganize_data(self):
        text_list, x_list, y_list, coords_list, time_info_list = [], [], [], [], []
        start_idx = self.train_num_time_steps + self.val_num_time_steps
        from tqdm import tqdm
        for i in tqdm(range(self.num_nodes), desc="Loading test nodes"):
            _, text, x, y, coords, time_info = self.original_dataset[i]
            text_list.append(text[:1])
            x_list.append(x[start_idx:])
            y_list.append(y[start_idx:])
            coords_list.append(coords)
            time_info_list.append({k: v[start_idx:] for k, v in time_info.items()})

        self.text_data = torch.stack(text_list, dim=1)
        self.x_samples_data = torch.stack(x_list, dim=1)
        if self.mask_flow_input:
            self.x_samples_data = torch.zeros_like(self.x_samples_data)
        self.y_samples_data = torch.stack(y_list, dim=1)
        self.coords_data = torch.stack(coords_list, dim=0)
        self.time_info_data = {k: torch.stack([time_info_list[i][k] for i in range(self.num_nodes)], dim=1) for k in
                               time_info_list[0].keys()}

        self.img_keys = []
        for t in range(self.test_num_time_steps):
            global_idx = start_idx + t
            self.img_keys.append([self.original_dataset.time_to_img_key[n][global_idx] for n in range(self.num_nodes)])

    def __len__(self):
        return self.test_num_time_steps

    def __getitem__(self, idx):
        keys = self.img_keys[idx]
        cache = getattr(self, 'image_cache', self.original_dataset.image_cache)
        dynamic_img_data = torch.stack([cache[k] for k in keys], dim=0)
        return (dynamic_img_data, self.text_data[0], self.x_samples_data[idx], self.y_samples_data[idx],
                self.coords_data, {k: v[idx] for k, v in self.time_info_data.items()})