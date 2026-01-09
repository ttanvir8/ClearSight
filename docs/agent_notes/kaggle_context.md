# Kaggle Training Context: ClearSight / WeatherDiffusion

## 1. Environment & Paths
*   **Platform**: Kaggle Notebooks
*   **Code Dataset location**: `/kaggle/input/clearsight`
*   **Data Dataset location**: `/kaggle/input/weather-data/data`
*   **Output/Working Directory**: `/kaggle/working`

## 2. Directory Structure (Kaggle)
The script expects the following structure in `/kaggle/input/weather-data/data`:
*   `allweather/`
    *   `input/` (Images)
    *   `gt/` (Ground Truth)
*   `allweather_val/`
    *   `input/`
    *   `gt/`

## 3. Critical Configuration Adjustments
To run successfully on Kaggle GPUs (P100/T4), the following changes were made to `configs/new_allweather.yml`:
*   **Batch Size**: Reduced from `16` to **`4`**.
    *   *Reason*: `patch_n` is 16, so effective batch size is `batch_size * patch_n`. $16 \times 16 = 256$ caused CUDA OutOfMemory. $4 \times 16 = 64$ fits in memory.
*   **Num Workers**: Reduced to **`2`**.
    *   *Reason*: Standard 32 workers cause shared memory errors in Kaggle/Docker containers.
*   **Paths**: All paths (`data_dir`, `train_data_dir`, etc.) must be updated to absolute paths within `/kaggle/working` or `/kaggle/input`.

## 4. Checkpoints & Outputs
*   **Location**: `/kaggle/working/ckpts/NewAllWeather_ddpm.pth.tar`
*   **Frequency**: Saved every `10,000` steps (controlled by `snapshot_freq`).
*   **Note**: Kaggle outputs inside directories must be downloaded manually or saved via "Save Version".

## 5. Complete Execution Script
Copy and run this script in a Kaggle Notebook cell. It handles importing code, generating file lists from the dataset, creating a compatible config file from scratch, and launching the training.

```python
import os
import sys
import yaml
import glob
import torch

# --- 1. Setup Paths ---
CODE_DIR = "/kaggle/input/clearsight"
DATA_ROOT = "/kaggle/input/weather-data/data"
WORK_DIR = "/kaggle/working"

sys.path.append(CODE_DIR)
print(f"Code directory added to path: {CODE_DIR}")

# --- 2. Generate File Lists ---
def generate_filelist(data_dir, output_filename):
    """
    Scans the 'input' subdirectory for images and creates a text file list.
    Entries will look like: input/image_name.jpg
    """
    input_dir = os.path.join(data_dir, "input")
    
    if not os.path.exists(input_dir):
        print(f"WARNING: Input directory not found: {input_dir}")
        return None, 0

    files = []
    # Search for common image formats
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    
    # Create relative paths (e.g., input/image.jpg)
    rel_files = [os.path.relpath(f, data_dir) for f in files]
    
    out_path = os.path.join(WORK_DIR, output_filename)
    with open(out_path, "w") as f:
        for line in rel_files:
            f.write(line + "\n")
            
    print(f"Generated {output_filename} with {len(rel_files)} images at {out_path}")
    return out_path, len(rel_files)

train_dir = os.path.join(DATA_ROOT, "allweather")
val_dir = os.path.join(DATA_ROOT, "allweather_val")

train_list_path, n_train = generate_filelist(train_dir, "train_files.txt")
val_list_path, n_val = generate_filelist(val_dir, "val_files.txt")

if n_train == 0:
    raise ValueError(f"No training images found in {os.path.join(train_dir, 'input')}!")

# --- 3. Create Custom Configuration ---
original_config_path = os.path.join(CODE_DIR, "configs", "new_allweather.yml")

with open(original_config_path, "r") as f:
    config = yaml.safe_load(f)

# === VITAL FIXES FOR KAGGLE ===
# 1. Update Paths to point to Kaggle directories
config['data']['data_dir'] = WORK_DIR 
config['data']['train_data_dir'] = train_dir
config['data']['val_data_dir'] = val_dir
config['data']['filelist'] = train_list_path
config['data']['val_filelist'] = val_list_path

# 2. Reduce Memory Usage (Fixing the OOM error)
# Previous: 16 (Resulting in 16*16=256 patches). New: 4 (Resulting in 4*16=64 patches)
config['training']['batch_size'] = 4  

# 3. Reduce Workers to avoid Docker shared memory limits
config['data']['num_workers'] = 2 

# Save new config
kaggle_config_path = os.path.join(WORK_DIR, "kaggle_train.yml")
with open(kaggle_config_path, "w") as f:
    yaml.dump(config, f)

print(f"Created Kaggle config at: {kaggle_config_path}")
print(f"Training Batch Size set to: {config['training']['batch_size']}")

# --- 4. Run Training ---
print("Starting Training...")
print("="*50)

# Cleaning CUDA cache before starting can sometimes help
torch.cuda.empty_cache()

script_path = os.path.join(CODE_DIR, "train_diffusion.py")
command = f"python {script_path} --config {kaggle_config_path}"

# Execute
exit_code = os.system(command)

if exit_code == 0:
    print("Training finished successfully!")
else:
    print("Training failed! Check the logs above.")
```

## 6. Local Setup and Relation
This section describes the setup on the local machine (Linux), distinct from Kaggle.

### Directory Structure
*   **Codebase**: `/home/tanvir/Work/754/project/other_code/ClearSight/`
    *   Contains `train_diffusion.py`, `configs/`, `datasets/`, `models/`, etc.
*   **Dataset Root**: `/home/tanvir/Work/754/project/other_code/WeatherDiffusion/`
    *   Acts as the `data_dir` in configuration.
    *   Contains `data/allweather` and `data/allweather_val`.
    *   Contains `ckpts/` (where checkoints are saved locally).

### Relation to Code
*   **Configuration**: `configs/new_allweather.yml` in the Codebase points to the absolute paths in the Dataset Root.
    *   `data_dir`: Points to `../WeatherDiffusion`
    *   `train_data_dir`: Points to `../WeatherDiffusion/data/allweather`
*   **Execution**:
    *   You run `python train_diffusion.py --config new_allweather.yml` from the `ClearSight` directory.
    *   The script reads the local config, which (unlike Kaggle) has valid absolute paths for the local machine.
    *   It's important that `new_allweather.yml` is **NOT** committed with Kaggle-specific paths, but kept with local paths. The Kaggle script (Section 5) generates a *temporary* config to handle the environment switch dynamically.
