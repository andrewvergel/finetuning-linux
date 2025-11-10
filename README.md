# Fine-tuning LoRA on Ubuntu + RTX 4060 Ti

> Personal project documented as part of my portfolio. Implements a complete fine-tuning pipeline with LoRA/QLoRA on modern models (Qwen2.5-7B-Instruct) using CUDA on Ubuntu, including environment setup, reproducible training, and inference deployment.

## ✨ Project Summary
- **Objective:** Train a corporate chatbot capable of responding to internal procedures from a custom instruction dataset.
- **Technologies:** Python, PyTorch 2.0+, Transformers 4.40+, PEFT 0.10+, TRL, CUDA 12.1+.
- **Hardware:** NVIDIA RTX 4060 Ti (16 GB VRAM) on Ubuntu 22.04+.
- **Key Features:**
  - Support for QLoRA 4-bit (memory optimization)
  - Stable training with bfloat16
  - Gradient Checkpointing and memory optimizations
  - Early Stopping and step-based evaluation
  - Optional sequence packing
- **Repository:** [`andrewvergel/finetuning-linux`](https://github.com/andrewvergel/finetuning-linux)

## 🧱 Architecture and Flow
1. **Environment bootstrap** – installation of drivers, CUDA and dependencies on a clean system.
2. **JSONL Dataset** – internal prompts versioned in `data/instructions.jsonl`.
3. **Training script** – `scripts/finetune_lora.py` (v2.0.0) uses structured configuration classes and performs intelligent dataset duplication and adjusts hyperparameters for low-data scenarios.
4. **Controlled inference** – `scripts/inference_lora.py` (v1.0.2) with deterministic decoding to evaluate results.
5. **Reports** – generates `training_info.json` with training metadata.

## 🚀 Getting Started from Scratch

### Quick Start (Local Development)

```bash
# 1. Clone the repository
git clone https://github.com/andrewvergel/finetuning-linux.git
cd finetuning-linux

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install base dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install "numpy<2.0" pyarrow==14.0.1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
pip install -r requirements.txt
```

### Server Deployment

This section provides step-by-step instructions for deploying and running the LoRA fine-tuning pipeline on a Linux server with NVIDIA GPU support.

#### Prerequisites

- Linux server (Ubuntu 22.04+ recommended)
- NVIDIA GPU with CUDA support (tested on RTX 4060 Ti 16GB)
- Python 3.10 or higher
- SSH access to the server
- Basic knowledge of Linux command line

#### Step 1: Initial Server Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential git curl wget

# Install NVIDIA drivers (for Ubuntu 22.04)
sudo apt install -y nvidia-driver-535

# Reboot to load drivers
sudo reboot

# After reboot, verify driver installation
nvidia-smi
```

#### Step 2: Install CUDA Toolkit

```bash
# Download CUDA 12.1 (or latest version)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Make executable and install
chmod +x cuda_12.1.0_530.30.02_linux.run
sudo ./cuda_12.1.0_530.30.02_linux.run

# Configure environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
```

#### Step 3: Set Up Python Environment

```bash
# Install Python 3.10+ if needed
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Create project directory
mkdir -p ~/projects/finetuning-lora
cd ~/projects/finetuning-lora

# Clone repository
git clone https://github.com/andrewvergel/finetuning-linux-cuda.git .

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch can see GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install project dependencies
pip install "numpy<2.0" pyarrow==14.0.1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
pip install -r requirements.txt
```

#### Step 4: Configure Environment

```bash
# Create .env file from example
cp env.example .env

# Edit .env file with your configuration
nano .env

# Set environment variables (optional, can also be in .env)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_MULTIPROCESSING=1
```

#### Step 5: Prepare Dataset

```bash
# Create data directory
mkdir -p data

# Create instructions.jsonl file (see Instruction Dataset section above)
nano data/instructions.jsonl

# Validate dataset format
python -c "
import json
with open('data/instructions.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line.strip())
            assert 'input' in data and 'output' in data
            print(f'Line {i}: OK')
        except Exception as e:
            print(f'Line {i}: ERROR - {e}')
"
```

#### Step 6: Validate Environment and Run Training

```bash
# Run environment validation script
python scripts/validate_environment.py

# Start training
source .venv/bin/activate
python scripts/finetune_lora.py

# Monitor training (in another terminal)
tail -f logs/debug_last_run.log
watch -n 1 nvidia-smi
```

#### Step 7: Run Training in Background (Optional)

```bash
# Run training in background with nohup
nohup python scripts/finetune_lora.py > training.out 2>&1 &

# Monitor output
tail -f training.out

# Check if process is running
ps aux | grep finetune_lora.py
```

#### Troubleshooting

**Out of Memory (OOM):**
- Reduce batch size: `FT_PER_DEVICE_BATCH_SIZE=1`
- Reduce sequence length: `FT_MAX_SEQ_LEN=256`
- Increase gradient accumulation: `FT_GRADIENT_ACCUMULATION=16`

**CUDA Out of Memory:**
- Enable QLoRA: `FT_USE_QLORA=true`
- Enable gradient checkpointing: `FT_GRADIENT_CHECKPOINTING=true`
- Disable packing: `FT_FORCE_PACKING=false`

**Slow Training:**
- Increase batch size: `FT_PER_DEVICE_BATCH_SIZE=2`
- Adjust gradient accumulation: `FT_GRADIENT_ACCUMULATION=4`

#### Troubleshooting Checklist

- [ ] NVIDIA drivers installed and working (`nvidia-smi` works)
- [ ] CUDA toolkit installed and in PATH (`nvcc --version` works)
- [ ] Python virtual environment activated
- [ ] PyTorch can see GPU (`torch.cuda.is_available()` returns `True`)
- [ ] All dependencies installed (`pip list` shows all packages)
- [ ] Dataset file exists and is valid (`data/instructions.jsonl`)
- [ ] `.env` file configured correctly
- [ ] Sufficient disk space available (`df -h`)
- [ ] Sufficient GPU memory available (`nvidia-smi`)
- [ ] Logs directory writable (`ls -ld logs/`)

> 💡 **New in v2.0.0:** The script has been refactored to use structured configuration classes (`ModelConfig`, `TrainingConfig`, `DataConfig`) for better code organization and maintainability.
> 📦 `requirements.txt` includes all auxiliary libraries; however, we install `numpy<2.0` and `pyarrow==14.0.1` first to avoid known conflicts with `datasets` (`PyExtensionType` error).
> 🔍 Before training, you can run `python scripts/validate_environment.py` to verify Python versions, CUDA, available VRAM, dataset and dependencies.
> 🧩 For Qwen2.5 models, make sure to use `transformers>=4.40` and `peft>=0.10` (already fixed in `requirements.txt`).
> 🧾 Each training run leaves a detailed log in `logs/debug_last_run.log` with metrics and automatic validation responses.

## 📚 Instruction Dataset
```bash
mkdir -p data
cat > data/instructions.jsonl << 'JSONL'
{"system":"You are an expert assistant in internal processes.","input":"Give me the steps to reconcile Monday payments.","output":"1) Export the CSV from the bank.\n2) Run the 'reconcile_monday' job.\n3) Review discrepancies in the 'recon_issues' table."}
{"system":"Speak in a professional and concise tone.","input":"Summarize this procedure in three bullets.","output":"• Export CSV.\n• Run job.\n• Validate discrepancies."}
{"system":"Always respond with numbered steps.","input":"How do I open a support ticket?","output":"1) Go to the helpdesk.\n2) Create 'Incident' ticket.\n3) Attach evidence."}
JSONL
```
> ℹ️ `data/instructions.jsonl` **is already versioned in this repository** and is the only file allowed inside `data/`. The training script automatically duplicates the dataset if it detects fewer than 200 samples, but it's recommended to manually expand it with more cases to improve response diversity.

## 🛠️ Training Script (`scripts/finetune_lora.py` v2.0.0)

### Changes in v2.0.0
- **Refactored to use structured configuration classes**: `ModelConfig`, `TrainingConfig`, `DataConfig`
- **Use of `ModelBuilder`**: Centralized and reusable model loading
- **Use of `DataProcessor`**: Centralized data processing
- **Better separation of concerns**: More maintainable and testable code
- **Maintains all previous functionality**: Compatible with existing configurations

### Main Features
- **Base Model:** `Qwen/Qwen2.5-7B-Instruct` by default (supports any Transformers-compatible model)
- **Memory Optimizations:**
  - QLoRA 4-bit activatable via `FT_USE_QLORA=1`
  - Gradient Checkpointing
  - bfloat16 by default (optimal for Ada/Lovelace GPUs)
  - Optional sequence packing (`FT_FORCE_PACKING`)
- **Stable Training:**
  - Early Stopping based on validation loss
  - Configurable step-based evaluation (`FT_EVAL_STEPS`)
  - Incremental checkpoint saving
  - Detailed logs in `logs/debug_last_run.log`
- **Flexible Configuration:**
  - All options configurable via `FT_*` environment variables
  - Support for multiple LoRA targets
  - Configurable batch size and gradient accumulation

### Training Flow
1. Load and validate dataset from `data/instructions.jsonl`
2. Automatic train/validation split (85/15% by default)
3. Load base model with memory optimizations
4. Apply LoRA/QLoRA according to configuration
5. Training with metric monitoring
6. Periodic evaluation and checkpoint saving
7. Generate final report with inference examples

### 🧾 Recommended Configuration (`.env`)

```bash
# Model and Data
FT_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
FT_DATA_PATH=data/instructions.jsonl
FT_OUT_DIR=models/out-qlora
FT_TRUST_REMOTE_CODE=1  # Required for Qwen2.5

# Memory Optimization
FT_USE_QLORA=1           # Enable QLoRA 4-bit
FT_FORCE_PACKING=0       # Disable packing by default (more memory)
FT_GRADIENT_CHECKPOINTING=1

# Training Hyperparameters
FT_PER_DEVICE_BATCH_SIZE=1
FT_GRADIENT_ACCUMULATION=8
FT_NUM_EPOCHS=5
FT_LEARNING_RATE=2e-5
FT_WARMUP_RATIO=0.1
FT_LR_SCHEDULER=cosine_with_restarts
FT_WEIGHT_DECAY=0.02

# LoRA Configuration
FT_LORA_RANK=8
FT_LORA_ALPHA=16
FT_LORA_DROPOUT=0.05
FT_LORA_TARGET_MODULES=q_proj,v_proj

# Validation and Saving
FT_EVAL_STEPS=25
FT_SAVE_STEPS=25
FT_SAVE_TOTAL_LIMIT=2
FT_EVAL_MAX_NEW_TOKENS=128
FT_EVAL_SAMPLE_SIZE=3

# Others
FT_LOGGING_STEPS=10
FT_DATASET_SHUFFLE_SEED=42
FT_VALIDATION_SPLIT=0.15
FT_DEBUG_LOG_FILE=debug_last_run.log
```
> Copy the file as `.env` and customize the values if you need to change any hyperparameter without editing the script.

### 🔍 Log Interpretation and Tuning
#### Training Loss
- `loss` values between **4.0–5.0** are typical for DialoGPT with repeated datasets. If it drops from ~5.2 to ~4.1 in a few epochs, convergence is going well. If loss plateaus >3.8 after 20 epochs, consider increasing `LEARNING_RATE` or reducing `LORA_DROPOUT`.
#### Effective Learning Rate
- With `LEARNING_RATE = 4e-5` you should see values ~3.9e-05 to 4.0e-05 in the logs. If it drops too quickly (<3e-05) in the first epochs, increase `WARMUP_RATIO`.
#### Overfitting Signals
- Low training loss, but validation doesn't improve or increases → increase `LORA_DROPOUT` or decrease `NUM_EPOCHS`.
#### If the Model Hallucinates
- Repeats phrases, circular or incoherent responses → increase `LORA_DROPOUT` to 0.2, decrease `NUM_EPOCHS` to 20, and add more unique examples to the dataset.

#### Advanced Training Phases (epochs 20+)
- If you see stable `loss` ~2.0 and `learning_rate` ~3e-06, the model is near the minimum. 
- Continuing training can lead to **subtle overfitting** (responds better to training examples but fails on variations). 
- **Stopping criterion:** if `eval_loss` stops decreasing for 3–4 consecutive epochs, stop training. 
- **If you need more quality:** instead of more epochs, expand the real dataset (don't repeat) or try a larger base model.

#### ✅ Healthy Progress Signals (epochs 1–5)
- `learning_rate` should rise from ~6e-06 to ~3e-05 during the first epochs (indicates warmup working).
- `eval_loss` should decrease consistently (e.g., 7.8 → 6.9 between epoch 1 and 3).
- Training `loss` between 7.0–8.5 at start, gradually decreasing.

### 🎯 Hyperparameter Tuning Guide

#### Memory Optimization (RTX 4060 Ti 16GB)
- **`FT_USE_QLORA` (1):** Enable 4-bit quantization (recommended for models >7B)
- **`FT_PER_DEVICE_BATCH_SIZE` (1):** Keep at 1 for maximum stability
- **`FT_GRADIENT_ACCUMULATION` (8):** Adjust according to available VRAM (higher = better GPU usage)
- **`FT_FORCE_PACKING` (0):** Disabled by default (uses more memory but more stable)

#### Training Performance
- **`FT_LORA_RANK` (8):** Dimension of low-rank matrices
  - *Increase* (16-32) for complex tasks
  - *Reduce* (4-8) if there are memory issues
- **`FT_LEARNING_RATE` (2e-5):** Base learning rate
  - *Increase* (3e-5) if loss plateaus
  - *Reduce* (1e-5) if loss is unstable
- **`FT_LORA_ALPHA` (16):** Scaling factor (typically 2× rank)

#### Regularization
- **`FT_LORA_DROPOUT` (0.05):** Regularization to avoid overfitting
  - *Increase* (0.1-0.2) if the model memorizes
  - *Reduce* (0.01) if learning is slow
- **`FT_WEIGHT_DECAY` (0.02):** Weight decay
  - *Increase* (0.05) for more regularization
  - *Reduce* (0.01) if the model doesn't converge

#### Evaluation
- **`FT_EVAL_STEPS` (25):** Evaluation frequency
- **`FT_EVAL_SAMPLE_SIZE` (3):** Number of examples for quick evaluation
- **`FT_EVAL_MAX_NEW_TOKENS` (128):** Maximum generation length in evaluation

## 💬 Inference Script (`scripts/inference_lora.py`)
- Loads the LoRA adapter from `models/out-tinyllama-lora`.
- Uses deterministic decoding (no sampling) to easily validate regressions.
- Includes optional interactive loop and GPU usage statistics.

## ▶️ Execution
```bash
# Training
python scripts/finetune_lora.py

# Initial inference
python scripts/inference_lora.py
```
> Training saves LoRA weights in `models/out-tinyllama-lora`. You can merge them with the base model using `scripts/merge_adapter.py` if you need a single checkpoint.

## 🧩 Project Structure
```
finetuning-linux/
├── data/
│   └── instructions.jsonl           # Versioned dataset
├── logs/
│   └── debug_last_run.log          # Detailed log of last training run
├── models/                          # Training outputs (ignored in git)
│   └── out-qlora/                  # Model checkpoints
│       ├── adapter_model.bin       # LoRA adapter weights
│       ├── config.json             # Model configuration
│       └── training_info.json      # Metrics and metadata
├── scripts/
│   ├── finetune_lora.py            # LoRA/QLoRA training (v1.2.0)
│   ├── inference_lora.py           # Inference with adapters
│   ├── merge_adapter.py            # Merge adapters with base model
│   └── validate_environment.py     # Environment verification
├── .env.example                    # Configuration template
├── .gitignore
└── README.md                       # This document
```
