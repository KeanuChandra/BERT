#!/bin/bash
#SBATCH --job-name=hubert_train
#SBATCH --account=fc_birdpow
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A5000:4
#SBATCH --time=12:00:00
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --output=train_%j.log

module load anaconda3
source activate hubert_env

echo "=============================================="
echo "HuBERT Training"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Verify GPU environment
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Number of GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

import torchaudio
print('torchaudio:', torchaudio.__version__)
print('Audio backends:', torchaudio.list_audio_backends())

import numpy
print('numpy:', numpy.__version__)

import scipy
print('scipy:', scipy.__version__)
"

echo ""
echo "Starting training..."
echo ""

# Training with spectrogram-based labels
srun python /global/home/users/jonathanswang/pytorchAudio/examples/hubert/train.py \
    --gpus 4 \
    --dataset-path /global/scratch/users/jonathanswang/temp_files/data/spectrogram/ \
    --exp-dir /global/scratch/users/jonathanswang/temp_train \
    --feature-type hubert \
    --dataset short_zebra_finch \
    --num-classes 100 \
    --max-updates 31250 \
    --learning-rate 0.0005

echo ""
echo "End time: $(date)"
echo "=============================================="
