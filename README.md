# ClearSight

Patch-based denoising diffusion models for weather-affected image restoration. This project restores images degraded by rain, fog, snow, and other weather conditions using diffusion-based approaches.

## Features

- Restore images affected by rain, fog, snow, and raindrops
- Patch-based diffusion model training and inference
- DDIM sampling for fast inference
- Automatic mixed precision (AMP) training support
- Multi-GPU training via DataParallel
- EMA (Exponential Moving Average) for improved model stability

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- PyTorch 1.12+
- torchvision
- numpy
- PyYAML
- tqdm
- Pillow
- opencv-python

## Project Structure

```
ClearSight/
├── train_diffusion.py     # Training script
├── eval_diffusion.py      # Evaluation/restoration script
├── configs/               # YAML configuration files
├── models/
│   ├── unet.py            # DiffusionUNet architecture
│   ├── ddm.py             # DenoisingDiffusion training class
│   └── restoration.py     # DiffusiveRestoration inference class
├── datasets/              # Dataset loaders
└── utils/                 # Utilities (logging, sampling, metrics)
```

## Usage

### Training

```bash
python train_diffusion.py --config new_allweather_toy_data.yml

# Resume from checkpoint
python train_diffusion.py --config new_allweather_toy_data.yml --resume path/to/checkpoint.pth.tar
```

### Evaluation

```bash
python eval_diffusion.py --config new_allweather_toy_data.yml --resume path/to/checkpoint.pth.tar

# With custom options
python eval_diffusion.py --config new_allweather_toy_data.yml --resume checkpoint.pth.tar \
    --sampling_timesteps 25 --grid_r 16 --test_set raindrop
```

### Arguments

**Training:**
- `--config`: Path to config file (required)
- `--resume`: Path to checkpoint for resuming training
- `--sampling_timesteps`: Number of DDIM sampling steps (default: 25)
- `--seed`: Random seed (default: 61)

**Evaluation:**
- `--config`: Path to config file (required)
- `--resume`: Path to trained model checkpoint (required)
- `--test_set`: Test dataset type (raindrop, snow, rainfog)
- `--grid_r`: Grid overlap size for patch-based restoration (default: 16)
- `--sampling_timesteps`: Number of DDIM sampling steps (default: 25)

## Configuration

Configurations are YAML files. Example structure:

```yaml
data:
  dataset: "ToyData"
  image_size: 64
  channels: 3

model:
  ch: 128
  ch_mult: [1, 2, 3, 4]
  num_res_blocks: 2
  attn_resolutions: [16]
  dropout: 0.0

diffusion:
  beta_schedule: linear
  beta_start: 0.0001
  beta_end: 0.02
  num_diffusion_timesteps: 1000

training:
  patch_n: 16
  batch_size: 1
  n_epochs: 1775

optim:
  optimizer: "Adam"
  lr: 0.00002
```

## Datasets

The project supports multiple dataset types:

1. **ToyData**: Small-scale dataset for testing
2. **NewAllWeather**: All-weather restoration dataset (rain, snow, fog)
3. **Fog**: Fog-specific degradation dataset

### Dataset Format

Datasets expect paired images with the following structure:
```
data_dir/
├── train.txt          # List of training image paths
├── val.txt            # List of validation image paths
├── input/             # Degraded images
│   └── xxx.jpg
└── gt/                # Ground truth images
    └── xxx.jpg
```

## Model Architecture

### DiffusionUNet

U-Net architecture with:
- Sinusoidal time embeddings
- ResNet blocks with group normalization
- Self-attention at specified resolutions
- Downsampling/upsampling with skip connections

### Training Details

- Noise estimation loss on random patches
- Antithetic time step sampling for variance reduction
- EMA for stable evaluation
- Mixed precision training with GradScaler

### Inference

- Patch-based image restoration with overlapping grids
- DDIM sampling for fast inference (25 steps default)
- Grid-based patch aggregation for seamless results

## Output

- Training checkpoints: `<data_dir>/ckpts/<dataset>_ddpm.pth.tar`
- Validation images: `results/images/<dataset><image_size>/<step>/`
- Restored images: `results/images/<dataset>/<test_set>/`

## Metrics

PSNR and SSIM metrics are available in `utils/metrics.py`:

```python
from utils.metrics import calculate_psnr, calculate_ssim

psnr = calculate_psnr(img1, img2)
ssim = calculate_ssim(img1, img2)
```

