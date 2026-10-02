import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import open_clip
from transformers import BertModel, BertTokenizer
from huggingface_hub import hf_hub_download
import shutil


def _safe_torch_load(path, map_location='cpu'):
    """
    Load torch checkpoint with backward-compatible API for different torch versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class RemoteCLIPEncoder(nn.Module):
    """
    RemoteCLIP encoder for remote sensing images.
    Specialized CLIP fine-tuned on satellite imagery.
    """
    def __init__(self, model_name='ViT-B-32', freeze=True, allow_offline_fallback=True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.allow_offline_fallback = allow_offline_fallback

        self._check_download_remoteclip()

        print(f"Loading RemoteCLIP-{model_name}...")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)

        self.is_pretrained_loaded = False
        if self.checkpoint_path is not None:
            try:
                ckpt = _safe_torch_load(self.checkpoint_path, map_location='cpu')
                model.load_state_dict(ckpt)
                self.is_pretrained_loaded = True
            except Exception as e:
                print(f"[WARN] Failed to load RemoteCLIP checkpoint: {e}")
                if not self.allow_offline_fallback:
                    raise
                print("[WARN] Falling back to randomly initialized RemoteCLIP visual encoder.")
                self.freeze = False
        else:
            if not self.allow_offline_fallback:
                raise FileNotFoundError(
                    f"RemoteCLIP-{model_name} checkpoint not found and fallback is disabled."
                )
            print("[WARN] RemoteCLIP checkpoint unavailable, using random initialization.")
            self.freeze = False
        
        self.visual = model.visual
        self.preprocess = preprocess
        
        output_dims = {
            'RN50': 1024,
            'ViT-B-32': 512,
            'ViT-L-14': 768
        }
        self.output_dim = output_dims[model_name]
        
        if self.freeze:
            for param in self.visual.parameters():
                param.requires_grad = False
            self.visual.eval()

        source = 'pretrained' if self.is_pretrained_loaded else 'random-init'
        print(f"RemoteCLIP-{model_name} loaded ({source}, output_dim={self.output_dim})")
    
    def _check_download_remoteclip(self):
        possible_paths = [
            f"checkpoints/RemoteCLIP-{self.model_name}.pt",
            f"checkpoints/models--chendelong--RemoteCLIP/RemoteCLIP-{self.model_name}.pt",
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found RemoteCLIP-{self.model_name} at: {path}")
                break

        is_offline = (
            os.getenv('SCENE2FLOW_OFFLINE', '0') == '1'
            or os.getenv('HF_HUB_OFFLINE', '0') == '1'
            or os.getenv('TRANSFORMERS_OFFLINE', '0') == '1'
        )
        
        if model_path is None and not is_offline:
            print(f"Downloading RemoteCLIP-{self.model_name}...")
            target_dir = 'checkpoints/models--chendelong--RemoteCLIP'
            os.makedirs(target_dir, exist_ok=True)
            model_path = f"{target_dir}/RemoteCLIP-{self.model_name}.pt"
            
            try:
                checkpoint_path = hf_hub_download(
                    "chendelong/RemoteCLIP",
                    f"RemoteCLIP-{self.model_name}.pt",
                    cache_dir='checkpoints'
                )
                shutil.copy(checkpoint_path, model_path)
                print(f"Downloaded to {model_path}")
            except Exception as e:
                print(f"Error downloading: {e}")
                print("Manual download:")
                print(f"   1. Download: https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-{self.model_name}.pt")
                print(f"   2. Place it at: checkpoints/RemoteCLIP-{self.model_name}.pt")
                model_path = None

        if model_path is None and is_offline:
            print("[INFO] Offline mode enabled and no usable local RemoteCLIP checkpoint was found.")
        
        self.checkpoint_path = model_path
    
    def forward(self, images):
        B, N, C, H, W = images.shape
        device = images.device
        
        images_flat = images.view(B * N, C, H, W)
        
        with torch.no_grad() if self.freeze else torch.enable_grad():
            img_features = self.visual(images_flat)
        
        img_features = img_features.view(B, N, -1).to(device)
        
        return img_features


class BERTEncoder(nn.Module):
    """
    BERT encoder for POI text.
    Returns all token embeddings for attention pooling.
    """
    def __init__(self, model_name='./checkpoints/bert-base-uncased', freeze=True, allow_offline_fallback=True):
        super().__init__()
        print(f"Loading {model_name} for text encoding...")
        self.allow_offline_fallback = allow_offline_fallback
        self.use_fallback = False

        is_offline = (
            os.getenv('SCENE2FLOW_OFFLINE', '0') == '1'
            or os.getenv('HF_HUB_OFFLINE', '0') == '1'
            or os.getenv('TRANSFORMERS_OFFLINE', '0') == '1'
        )
        is_local_path = model_name.startswith('.') or model_name.startswith('/')
        local_missing = is_local_path and (not os.path.exists(model_name))
        # If a local path is provided (e.g. ./checkpoints/bert-base-uncased),
        # derive a remote model id fallback (bert-base-uncased) for online loading.
        remote_model_id = os.path.basename(os.path.normpath(model_name)) if is_local_path else model_name
        if not remote_model_id:
            remote_model_id = 'bert-base-uncased'

        self.bert = None
        self.tokenizer = None

        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )
            self.bert = BertModel.from_pretrained(
                model_name,
                local_files_only=True
            )
            print(f"✓ Loaded {model_name} from local cache")
        except Exception as local_error:
            if not is_offline:
                candidate_model_ids = [remote_model_id]
                if remote_model_id == 'bert-base-uncased':
                    candidate_model_ids.append('google-bert/bert-base-uncased')

                print(f"[INFO] Trying online BERT load: {candidate_model_ids}")
                print("  (This is a one-time download and will be cached under checkpoints/)")

                def _try_download(model_id: str) -> bool:
                    try:
                        self.tokenizer = BertTokenizer.from_pretrained(
                            model_id,
                            cache_dir='checkpoints'
                        )
                        self.bert = BertModel.from_pretrained(
                            model_id,
                            cache_dir='checkpoints'
                        )
                        print(f"✓ Downloaded and loaded {model_id}")
                        return True
                    except Exception as err:
                        print(f"[WARN] Failed to download BERT ({model_id}): {err}")
                        return False

                loaded = False
                for candidate in candidate_model_ids:
                    if _try_download(candidate):
                        remote_model_id = candidate
                        loaded = True
                        break

                # Retry with HF mirror when direct access is blocked.
                if (not loaded) and (os.getenv('HF_ENDPOINT') is None):
                    previous_endpoint = os.environ.get('HF_ENDPOINT')
                    try:
                        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                        print("[INFO] Retrying with HF mirror endpoint: https://hf-mirror.com")
                        for candidate in candidate_model_ids:
                            if _try_download(candidate):
                                remote_model_id = candidate
                                loaded = True
                                break
                    finally:
                        if previous_endpoint is None:
                            os.environ.pop('HF_ENDPOINT', None)
                        else:
                            os.environ['HF_ENDPOINT'] = previous_endpoint

                if (not loaded) and local_missing:
                    print(f"[INFO] Local BERT path not found: {model_name}.")
            else:
                print("[INFO] Offline mode enabled; skip online BERT download.")

            if (self.bert is not None and self.tokenizer is not None) and is_local_path:
                try:
                    os.makedirs(model_name, exist_ok=True)
                    self.tokenizer.save_pretrained(model_name)
                    self.bert.save_pretrained(model_name)
                    print(f"✓ Saved downloaded BERT to local path: {model_name}")
                except Exception as save_error:
                    print(f"[WARN] Failed to save downloaded BERT to {model_name}: {save_error}")

            if self.bert is None or self.tokenizer is None:
                if not self.allow_offline_fallback:
                    raise local_error
                self.use_fallback = True
        
        self.output_dim = 768
        self.freeze = freeze
        
        if self.use_fallback:
            self.fallback_embedding = nn.Embedding(512, self.output_dim)
            self.freeze = False
            print("[WARN] BERT fallback active: using trainable embedding encoder.")
        elif freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()
        
        print(f"BERT loaded (output_dim={self.output_dim})")

    
    def forward(self, input_ids, attention_mask=None, return_all_tokens=True):
        B, N, L = input_ids.shape
        
        input_ids_flat = input_ids.view(B * N, L)
        
        if attention_mask is not None:
            attention_mask_flat = attention_mask.view(B * N, L)
        else:
            # Default to padding-aware mask for pre-tokenized BERT ids.
            attention_mask_flat = (input_ids_flat != 0).long()
        
        if self.use_fallback:
            input_ids_safe = input_ids_flat.long().clamp(0, self.fallback_embedding.num_embeddings - 1)
            last_hidden_state = self.fallback_embedding(input_ids_safe)
        else:
            if self.freeze:
                with torch.no_grad():
                    outputs = self.bert(
                        input_ids=input_ids_flat,
                        attention_mask=attention_mask_flat
                    )
            else:
                outputs = self.bert(
                    input_ids=input_ids_flat,
                    attention_mask=attention_mask_flat
                )
            last_hidden_state = outputs.last_hidden_state
        
        if return_all_tokens:
            embeddings = last_hidden_state.view(B, N, L, -1)
            mask = attention_mask_flat.view(B, N, L).bool()
            return embeddings, mask
        else:
            cls_embeddings = last_hidden_state[:, 0, :]
            embeddings = cls_embeddings.view(B, N, -1)
            return embeddings


class FeatureExtractionModule(nn.Module):
    """
    Combined feature extraction module.
    Integrates RemoteCLIP and BERT with trainable projections.
    """
    def __init__(self, use_remoteclip=True, remoteclip_model='ViT-B-32', image_token_count=8,
                 allow_offline_fallback=True):
        super().__init__()
        self.use_remoteclip = use_remoteclip
        self.image_token_count = max(1, int(image_token_count))
        
        if use_remoteclip:
            self.img_encoder = RemoteCLIPEncoder(
                model_name=remoteclip_model,
                freeze=True,
                allow_offline_fallback=allow_offline_fallback
            )
            self.img_output_dim = self.img_encoder.output_dim
        else:
            raise NotImplementedError("Standard CLIP fallback not implemented")
        
        self.img_proj = nn.Linear(self.img_output_dim, self.img_output_dim)
        self.img_token_proj = nn.Linear(
            self.img_output_dim,
            self.img_output_dim * self.image_token_count
        )
        
        self.text_encoder = BERTEncoder(
            freeze=True,
            allow_offline_fallback=allow_offline_fallback
        )
        self.text_output_dim = self.text_encoder.output_dim
        
        self.text_proj = nn.Linear(self.text_output_dim, self.text_output_dim)

    def freeze_pretrained_backbones(self):
        if hasattr(self, 'img_encoder'):
            if getattr(self.img_encoder, 'is_pretrained_loaded', False):
                self.img_encoder.eval()
                for param in self.img_encoder.parameters():
                    param.requires_grad = False
        if hasattr(self, 'text_encoder'):
            if not getattr(self.text_encoder, 'use_fallback', False):
                self.text_encoder.eval()
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
    
    def encode_img(self, images, return_tokens=False):
        img_features = self.img_encoder(images)
        img_features = self.img_proj(img_features)
        if not return_tokens:
            return img_features

        B, N, D = img_features.shape
        token_bank = self.img_token_proj(img_features).view(B, N, self.image_token_count, D)
        return img_features, token_bank
    
    def encode_text(self, input_ids, attention_mask=None, return_all_tokens=True):
        if return_all_tokens:
            text_features, mask = self.text_encoder(
                input_ids, attention_mask, return_all_tokens=True
            )
            B, N, L, D = text_features.shape
            text_features_flat = text_features.view(B * N * L, D)
            text_features_proj = self.text_proj(text_features_flat)
            text_features = text_features_proj.view(B, N, L, D)
            return text_features, mask
        else:
            text_features = self.text_encoder(
                input_ids, attention_mask, return_all_tokens=False
            )
            text_features = self.text_proj(text_features)
            return text_features