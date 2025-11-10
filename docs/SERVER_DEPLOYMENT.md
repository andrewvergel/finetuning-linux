# Server Deployment Guide: Step-by-Step Instructions

This guide provides detailed step-by-step instructions for deploying and running the LoRA fine-tuning pipeline on a Linux server with NVIDIA GPU support.

## Prerequisites

- Linux server (Ubuntu 22.04+ recommended)
- NVIDIA GPU with CUDA support (tested on RTX 4060 Ti 16GB)
- Python 3.10 or higher
- SSH access to the server
- Basic knowledge of Linux command line

## Step 1: Initial Server Setup

### 1.1 Update System Packages

```bash
# Update package list
sudo apt update

# Upgrade existing packages
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential git curl wget
```

### 1.2 Install NVIDIA Drivers

```bash
# Check if NVIDIA GPU is detected
lspci | grep -i nvidia

# Install NVIDIA drivers (for Ubuntu 22.04)
sudo apt install -y nvidia-driver-535

# Alternative: Use NVIDIA's official repository
# Visit https://developer.nvidia.com/cuda-downloads for latest drivers

# Reboot to load drivers
sudo reboot
```

### 1.3 Verify NVIDIA Installation

```bash
# After reboot, verify driver installation
nvidia-smi

# You should see GPU information and CUDA version
# Expected output shows GPU model, memory, and CUDA version
```

## Step 2: Install CUDA Toolkit

### 2.1 Download and Install CUDA 12.1+

```bash
# Download CUDA 12.1 (or latest version)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Make executable
chmod +x cuda_12.1.0_530.30.02_linux.run

# Install CUDA (accept license, keep default options)
sudo ./cuda_12.1.0_530.30.02_linux.run
```

### 2.2 Configure Environment Variables

```bash
# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reload shell configuration
source ~/.bashrc

# Verify CUDA installation
nvcc --version
```

## Step 3: Install Python and Virtual Environment

### 3.1 Install Python 3.10+

```bash
# Check Python version
python3 --version

# If Python 3.10+ is not installed, install it
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Install pip and setuptools
python3 -m pip install --upgrade pip setuptools wheel
```

### 3.2 Create Project Directory

```bash
# Create project directory
mkdir -p ~/projects/finetuning-lora
cd ~/projects/finetuning-lora

# Clone repository (if using git)
# git clone https://github.com/andrewvergel/finetuning-linux-cuda.git .
# OR create directory structure manually
```

## Step 4: Set Up Python Virtual Environment

### 4.1 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation (prompt should show (.venv))
which python
```

### 4.2 Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch can see GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 4.3 Install Project Dependencies

```bash
# Install numpy and pyarrow first (to avoid compatibility issues)
pip install "numpy<2.0" pyarrow==14.0.1

# Install project requirements
pip install -r requirements.txt

# Verify critical packages
python -c "import transformers; import peft; import trl; print('All packages installed successfully')"
```

## Step 5: Configure Environment Variables

### 5.1 Create .env File

```bash
# Create .env file from example
cp env.example .env

# Edit .env file with your configuration
nano .env
```

### 5.2 Configure .env File

```bash
# ===== Model Configuration =====
FT_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
FT_USE_QLORA=true
FT_TRUST_REMOTE_CODE=true

# ===== Data Configuration =====
FT_DATA_PATH=data/instructions.jsonl
FT_OUT_DIR=models/out-qlora

# ===== Training Configuration =====
FT_PER_DEVICE_BATCH_SIZE=1
FT_GRADIENT_ACCUMULATION=8
FT_NUM_EPOCHS=8
FT_LEARNING_RATE=1.5e-5
FT_WARMUP_RATIO=0.1
FT_LR_SCHEDULER=cosine_with_restarts
FT_WEIGHT_DECAY=0.02

# ===== LoRA Configuration =====
FT_LORA_RANK=16
FT_LORA_ALPHA=16
FT_LORA_DROPOUT=0.05
FT_LORA_TARGET_MODULES=q_proj,v_proj

# ===== Memory Optimization =====
FT_MAX_SEQ_LEN=512
FT_FORCE_PACKING=false
FT_GRADIENT_CHECKPOINTING=true

# ===== Evaluation Configuration =====
FT_EVAL_STEPS=25
FT_SAVE_STEPS=25
FT_SAVE_TOTAL_LIMIT=2
FT_EVAL_MAX_NEW_TOKENS=128
FT_EVAL_SAMPLE_SIZE=3

# ===== Logging Configuration =====
FT_LOGGING_STEPS=10
FT_DATASET_SHUFFLE_SEED=42
FT_VALIDATION_SPLIT=0.15
FT_DEBUG_LOG_FILE=debug_last_run.log

# ===== PyTorch Memory Configuration =====
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
CUDA_LAUNCH_BLOCKING=1
TOKENIZERS_PARALLELISM=false
HF_DATASETS_DISABLE_MULTIPROCESSING=1
```

### 5.3 Set Environment Variables (Optional)

```bash
# Export environment variables for current session
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_MULTIPROCESSING=1
```

## Step 6: Prepare Dataset

### 6.1 Create Data Directory

```bash
# Create data directory
mkdir -p data

# Create instructions.jsonl file
nano data/instructions.jsonl
```

### 6.2 Add Training Data

```jsonl
{"system":"Eres un asistente experto en procesos internos.","input":"Dame los pasos para conciliar pagos de los lunes.","output":"1) Exporta el CSV del banco.\n2) Ejecuta el job 'reconcile_monday'.\n3) Revisa discrepancias en la tabla 'recon_issues'."}
{"system":"Habla en tono profesional y conciso.","input":"Resume este procedimiento en tres bullets.","output":"• Exportar CSV.\n• Ejecutar job.\n• Validar discrepancias."}
{"system":"Responde siempre con pasos numerados.","input":"¿Cómo abro un ticket de soporte?","output":"1) Entra al helpdesk.\n2) Crea ticket 'Incidente'.\n3) Adjunta evidencias."}
```

### 6.3 Validate Dataset

```bash
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

## Step 7: Validate Environment

### 7.1 Run Environment Validation Script

```bash
# Run validation script
python scripts/validate_environment.py

# Expected output:
# - Python version check
# - CUDA availability
# - VRAM information
# - Dataset validation
# - Package versions
```

### 7.2 Verify GPU Memory

```bash
# Check GPU memory
nvidia-smi

# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

## Step 8: Run Training

### 8.1 Start Training

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Run training script
python scripts/finetune_lora.py

# Training will:
# - Load model and tokenizer
# - Process dataset
# - Train with LoRA/QLoRA
# - Save checkpoints
# - Generate evaluation metrics
```

### 8.2 Monitor Training

```bash
# Monitor training logs
tail -f logs/debug_last_run.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training progress
ls -lh models/out-qlora/
```

### 8.3 Run Training in Background (Optional)

```bash
# Run training in background with nohup
nohup python scripts/finetune_lora.py > training.out 2>&1 &

# Monitor output
tail -f training.out

# Check if process is running
ps aux | grep finetune_lora.py
```

## Step 9: Monitor and Troubleshoot

### 9.1 Check Training Logs

```bash
# View latest log file
tail -f logs/debug_last_run.log

# Search for errors
grep -i error logs/debug_last_run.log

# Check training metrics
grep "train_loss" logs/debug_last_run.log
```

### 9.2 Monitor System Resources

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor CPU and memory
htop

# Monitor disk usage
df -h
```

### 9.3 Common Issues and Solutions

#### Issue: Out of Memory (OOM)

```bash
# Solution: Reduce batch size or sequence length
# Edit .env file:
FT_PER_DEVICE_BATCH_SIZE=1
FT_MAX_SEQ_LEN=256
FT_GRADIENT_ACCUMULATION=16
```

#### Issue: CUDA Out of Memory

```bash
# Solution: Enable QLoRA and reduce memory usage
# Edit .env file:
FT_USE_QLORA=true
FT_FORCE_PACKING=false
FT_GRADIENT_CHECKPOINTING=true
```

#### Issue: Slow Training

```bash
# Solution: Optimize configuration
# Edit .env file:
FT_PER_DEVICE_BATCH_SIZE=2
FT_GRADIENT_ACCUMULATION=4
FT_DATASET_NUM_PROC=4
```

## Step 10: Post-Training Tasks

### 10.1 Check Training Results

```bash
# View training metrics
cat models/out-qlora/training_info.json

# Check saved model
ls -lh models/out-qlora/

# View training logs
cat logs/debug_last_run.log | grep "Métricas de evaluación"
```

### 10.2 Run Inference

```bash
# Run inference script
python scripts/inference_lora.py

# Test model with sample prompts
python -c "
from scripts.inference_lora import load_model_and_tokenizer
model, tokenizer = load_model_and_tokenizer('models/out-qlora')
# Add inference code here
"
```

### 10.3 Save and Backup Model

```bash
# Create backup directory
mkdir -p ~/backups/models

# Copy trained model
cp -r models/out-qlora ~/backups/models/$(date +%Y%m%d_%H%M%S)

# Compress for storage
tar -czf ~/backups/models/qlora_model_$(date +%Y%m%d).tar.gz models/out-qlora/
```

## Step 11: System Optimization (Optional)

### 11.1 Enable Persistence Mode

```bash
# Enable GPU persistence mode (keeps GPU initialized)
sudo nvidia-smi -pm 1
```

### 11.2 Set GPU Performance Mode

```bash
# Set GPU to maximum performance
sudo nvidia-smi -ac <memory_clock>,<graphics_clock>

# Example for RTX 4060 Ti:
# sudo nvidia-smi -ac 8001,2505
```

### 11.3 Configure Swap (if needed)

```bash
# Check swap usage
free -h

# Create swap file if needed (8GB example)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Step 12: Automation and Scheduling

### 12.1 Create Training Script

```bash
# Create training script
cat > ~/train_model.sh << 'EOF'
#!/bin/bash
cd ~/projects/finetuning-lora
source .venv/bin/activate
python scripts/finetune_lora.py
EOF

# Make executable
chmod +x ~/train_model.sh
```

### 12.2 Schedule Training with Cron

```bash
# Edit crontab
crontab -e

# Add entry to run training daily at 2 AM
# 0 2 * * * /home/user/train_model.sh >> /home/user/training.log 2>&1
```

### 12.3 Use systemd Service (Optional)

```bash
# Create systemd service file
sudo nano /etc/systemd/system/finetuning.service

# Add service configuration:
[Unit]
Description=LoRA Fine-tuning Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/projects/finetuning-lora
ExecStart=/home/your_username/projects/finetuning-lora/.venv/bin/python scripts/finetune_lora.py
Restart=on-failure

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl enable finetuning.service
sudo systemctl start finetuning.service
```

## Troubleshooting Checklist

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

## Additional Resources

- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

## Support

For issues or questions:
1. Check logs in `logs/debug_last_run.log`
2. Verify environment with `scripts/validate_environment.py`
3. Review configuration in `.env` file
4. Check GPU status with `nvidia-smi`

## Notes

- Training time depends on dataset size, model size, and hardware
- Monitor GPU temperature and usage during training
- Regular backups recommended for trained models
- Consider using screen or tmux for long-running training sessions

