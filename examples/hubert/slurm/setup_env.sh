#!/bin/bash
#SBATCH --job-name=setup_hubert_env
#SBATCH --account=fc_birdpow
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=setup_env_%j.log

module load anaconda3

# Fix for NFS filesystem locking issues on Savio
export CONDA_NO_LOCK=1
export CONDA_PKGS_DIRS="/tmp/${USER}/conda_pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

ENV_NAME="hubert_env"

echo "=============================================="
echo "Setting up HuBERT training environment: $ENV_NAME"
echo "=============================================="

# OPTIONAL: remove old env if needed (uncomment to recreate from scratch)
# conda remove --name $ENV_NAME --all -y

# Create environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating new conda environment: $ENV_NAME"
    conda create --name $ENV_NAME python=3.11 -y
else
    echo "Environment $ENV_NAME already exists, updating..."
fi

source activate $ENV_NAME

echo ""
echo "=== Installing PyTorch with CUDA support ==="
conda install pytorch=2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo ""
echo "=== Installing PyTorch Lightning ==="
# Using pip instead of conda - more reliable for lightning
pip install lightning torchmetrics

echo ""
echo "=== Installing numerical libraries ==="
conda install numpy=1.26 scipy -y

echo ""
echo "=== Installing scikit-learn (for KMeans clustering) ==="
conda install -c conda-forge scikit-learn -y

echo ""
echo "=== Installing joblib (for parallel processing) ==="
conda install -c conda-forge joblib -y

echo ""
echo "=== Installing audio processing libraries ==="
conda install -c conda-forge 'ffmpeg<7' -y

echo ""
echo "=== Installing soundsig (for spectrogram extraction) ==="
pip install soundsig

echo ""
echo "=== Patching soundsig for scipy compatibility ==="
python /global/home/users/jonathanswang/pytorchAudio/examples/hubert/slurm/fix_soundsig.py

echo ""
echo "=== Installing visualization libraries (optional) ==="
conda install -c conda-forge matplotlib seaborn -y

echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import torchaudio
print(f'torchaudio: {torchaudio.__version__}')
print(f'Audio backends: {torchaudio.list_audio_backends()}')

import lightning
print(f'Lightning: {lightning.__version__}')

import numpy
print(f'NumPy: {numpy.__version__}')

import scipy
print(f'SciPy: {scipy.__version__}')

import sklearn
print(f'scikit-learn: {sklearn.__version__}')

try:
    from soundsig.sound import spectrogram
    print('soundsig: OK')
except Exception as e:
    print(f'soundsig: FAILED - {e}')

print()
print('✅ Environment setup complete!')
"

echo ""
echo "=============================================="
echo "To use this environment, run:"
echo "  source activate $ENV_NAME"
echo "=============================================="
