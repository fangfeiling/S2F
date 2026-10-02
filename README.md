# Sence2Flow

Sence2Flow is a deep learning framework for multi-modal traffic prediction, integrating traffic flow data with Points of Interest (POI), POI text descriptions, and Remote Sensing (RS) imagery.

## Project Structure

The project consists of two main directories: the source code (`Sence2Flow`) and the dataset directory (`data`).

```text
.
├── data/                       # Dataset Root
│   ├── BM/                     # Beijing Data
│   │   ├── POI/             # POI Information
│   │   ├── POI_TEXT/        # Textual descriptions of POIs
│   │   ├── RS/              # Remote Sensing Imagery
│   │   ├── TAXI/            # Traffic Flow Data
│   │   └── geo.csv             # Geographic coordinates
│   ├── NYC/                    # New York City Data
│   └── DC/                     # San Francisco Data
│
└── Sence2Flow/                 # Source Code Root
    ├── checkpoints/            # Pretrained models & saved weights
    ├── config/                 # Configuration files (*.conf)
    ├── layers/                 # Neural network layers
    ├── models/                 # Main model definitions
    ├── scripts/                # Bash scripts (run_*_v2.sh, eval_6models_in_transfer.sh)
    ├── utils/                  # Utility functions (metrics, loaders)
    ├── preprocess.py           # Data preprocessing script
    ├── train.py                # Training script
    ├── test.py                 # Evaluation script
    └── requirements.txt        # Python dependencies
```

## Environment Setup

Ensure you have the necessary Python dependencies installed:

```bash
pip install -r requirements.txt
```

## Pretrained Models (BERT & RemoteCLIP)

Sence2Flow uses pretrained models for text (BERT) and image (RemoteCLIP) encoding. The code is designed to handle downloads automatically, but manual setup is supported for offline environments.

### 1. Automatic Download
On the first run, the script attempts to download weights automatically via HuggingFace:
*   **BERT**: Loads `bert-base-uncased` using the `transformers` library.
*   **RemoteCLIP**: Downloads `RemoteCLIP-ViT-B-32.pt` to the `checkpoints/` folder.

### 2. Manual Setup (Offline Mode)
If the automatic download fails or the server lacks internet access, follow these steps:

**RemoteCLIP:**
1. Download the weight file: [RemoteCLIP-ViT-B-32.pt](https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt)
2. Place it at: `Sence2Flow/checkpoints/RemoteCLIP-ViT-B-32.pt`

**BERT:**
1. Download the `bert-base-uncased` model files (config, binaries, vocab, etc.).
2. Place the directory at: `Sence2Flow/checkpoints/bert-base-uncased/`

## 1. Data Preprocessing

The `preprocess.py` script reads the raw data paths defined in the configuration files (`config/*.conf`) and generates cached pickle files.

This step generates `train_dataset.pkl`, `val_dataset.pkl`, `test_dataset.pkl`, and `base_info.pkl` inside the `preprocessed_data/<DATA>/` directory. While the first run may take time, subsequent training and testing will load data instantly using the cache.

Run the following commands to preprocess data for the three cities (v2 cache + precomputed embeddings):

```bash
python preprocess.py --config config/BM.conf
python preprocess.py --config config/NYC.conf
python preprocess.py --config config/DC.conf
```

**Directory structure after preprocessing:**

```text
preprocessed_data/
├── BM_TAXI/
│   ├── train_dataset.pkl
│   ├── val_dataset.pkl
│   ├── test_dataset.pkl
│   ├── base_info.pkl
│   ├── image_embedding_cache.pt
│   └── text_embedding_cache.pt
├── NYC_TAXI/
└── DC_TAXI/
```

## 2. Training

To train the model, use `train.py`. v2 training always reads `preprocessed_data/<DATA>_v2`. Below is an example for NYC.

```bash
python train.py \
  --config config/NYC.conf \
  --save_dir checkpoints/NYC/ \
  --device cuda
```

**Common Arguments:**
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--lr`: Learning rate.
- `--mask_flow_input`: Option to mask historical flow input for specific training strategies.

## 3. Inference / Testing

Use `test.py` to evaluate the trained model weights. This script calculates metrics and generates prediction files (`.npy`) for visualization.

```bash
python test.py \
  --config config/BM.conf \
  --checkpoint checkpoints/BM/best_model.pth \
  --output_dir results/BM/ \
  --save_predictions \
  --device cuda \
  --batch_size 16

python test.py \
  --config config/DC.conf \
  --checkpoint checkpoints/DC/best_model.pth \
  --output_dir results/DC/ \
  --save_predictions \
  --device cuda \
  --batch_size 16
```

**Output Files:**
- `results/NYC/test_metrics.json`: Contains evaluation metrics (MAE, RMSE, MAPE, etc.).
- `results/NY/predictions.npy`: Model prediction values.
- `results/NYC/targets.npy`: Ground truth values.
