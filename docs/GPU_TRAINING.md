# GPU Training Guide

This guide covers training the BERTić model on GPU for faster performance.

## Training on AMD GPU (ROCm)

### 1. System Requirements

- AMD GPU with ROCm support (RX 6000/7000 series, Radeon Pro, Instinct)
- Linux (ROCm doesn't support Windows well)
- ROCm 5.4+ installed

### 2. Check ROCm Installation

```bash
# Verify ROCm is installed
rocm-smi

# Should show your GPU info
```

If ROCm isn't installed, see: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html

### 3. Clone and Setup

```bash
# Clone the repo
git clone https://github.com/TeoMatosevic/slur-analysis-model.git
cd slur-analysis-model

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install other dependencies
pip install -r requirements.txt

# Download Croatian NLP models
python -c "import classla; classla.download('hr', type='nonstandard')"
```

### 4. Verify GPU is Detected

```bash
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Should output:
```
ROCm available: True
GPU: AMD Radeon RX 7900 XTX  (or your GPU name)
```

### 5. Configure Training Epochs

Edit `configs/config.yaml`:

```yaml
# Find this section and change num_epochs:
bertic:
  training:
    learning_rate: 2.0e-5
    batch_size: 16
    num_epochs: 5      # Default: 5 (3-5 recommended)
    warmup_ratio: 0.1
    weight_decay: 0.01
    max_length: 256
```

### 6. Run Training

```bash
# Train BERTić model
python src/training/train.py \
    --data data/processed/frenk_train.jsonl \
    --model bertic \
    --output checkpoints/bertic

# Or train all models (baseline + BERTić)
python src/training/train.py \
    --data data/processed/frenk_train.jsonl \
    --model all \
    --output checkpoints
```

### 7. Expected Performance

| Hardware | Time per Epoch | 5 Epochs Total |
|----------|---------------|----------------|
| CPU (Intel i7) | ~1.7 hours | ~8.5 hours |
| AMD RX 6800 | ~8-10 min | ~40-50 min |
| AMD RX 7900 XTX | ~4-6 min | ~20-30 min |
| AMD MI100/MI250 | ~2-3 min | ~10-15 min |

### 8. Monitor Training

```bash
# Watch GPU usage
watch -n 1 rocm-smi

# In another terminal, watch training progress
tail -f checkpoints/bertic/training.log
```

### 9. Evaluate Results

```bash
# After training completes
python src/training/evaluate.py \
    --data data/processed/frenk_test.jsonl \
    --model bertic \
    --model-path checkpoints/bertic/best_model \
    --output evaluation_results
```

---

## Training on NVIDIA GPU (CUDA)

### 1. System Requirements

- NVIDIA GPU with CUDA support (GTX 1000+, RTX series, Tesla, etc.)
- CUDA 11.8+ and cuDNN installed
- Linux or Windows

### 2. Setup

```bash
# Clone the repo
git clone https://github.com/TeoMatosevic/slur-analysis-model.git
cd slur-analysis-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Or for CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Download Croatian NLP models
python -c "import classla; classla.download('hr', type='nonstandard')"
```

### 3. Verify GPU is Detected

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 4. Run Training

Same as AMD - edit `configs/config.yaml` to set epochs, then:

```bash
python src/training/train.py \
    --data data/processed/frenk_train.jsonl \
    --model bertic \
    --output checkpoints/bertic
```

### 5. Expected Performance (NVIDIA)

| Hardware | Time per Epoch | 5 Epochs Total |
|----------|---------------|----------------|
| GTX 1080 | ~8-10 min | ~40-50 min |
| RTX 3080 | ~4-5 min | ~20-25 min |
| RTX 4090 | ~2-3 min | ~10-15 min |
| A100 | ~1-2 min | ~5-10 min |

---

## Troubleshooting

### ROCm: "ROCm not detected"

```bash
# Check if ROCm kernel module is loaded
lsmod | grep amdgpu

# Check ROCm version
cat /opt/rocm/.info/version

# Make sure user is in video/render groups
sudo usermod -a -G video,render $USER
# Then logout and login again
```

### CUDA: "CUDA not available"

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory (OOM)

Reduce batch size in `configs/config.yaml`:

```yaml
bertic:
  training:
    batch_size: 8    # Reduce from 16 to 8
    # Or even 4 for GPUs with <8GB VRAM
```

### PyTorch Doesn't See GPU

```bash
# For AMD (ROCm)
export HIP_VISIBLE_DEVICES=0

# For NVIDIA (CUDA)
export CUDA_VISIBLE_DEVICES=0
```

---

## Quick Reference

### AMD ROCm Setup (One Command)

```bash
git clone https://github.com/TeoMatosevic/slur-analysis-model.git && cd slur-analysis-model && \
python -m venv venv && source venv/bin/activate && \
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0 && \
pip install -r requirements.txt && \
python -c "import classla; classla.download('hr', type='nonstandard')" && \
python src/training/train.py --data data/processed/frenk_train.jsonl --model bertic --output checkpoints/bertic
```

### NVIDIA CUDA Setup (One Command)

```bash
git clone https://github.com/TeoMatosevic/slur-analysis-model.git && cd slur-analysis-model && \
python -m venv venv && source venv/bin/activate && \
pip install torch --index-url https://download.pytorch.org/whl/cu118 && \
pip install -r requirements.txt && \
python -c "import classla; classla.download('hr', type='nonstandard')" && \
python src/training/train.py --data data/processed/frenk_train.jsonl --model bertic --output checkpoints/bertic
```

---

## Continue Training from Checkpoint

If training was interrupted or you want to train more epochs:

```bash
# Load existing model and continue training
python src/training/train.py \
    --data data/processed/frenk_train.jsonl \
    --model checkpoints/bertic/best_model \
    --output checkpoints/bertic_continued
```
