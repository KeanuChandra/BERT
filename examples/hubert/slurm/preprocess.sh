#!/bin/bash
#SBATCH --job-name=hubert_preprocess
#SBATCH --account=fc_birdpow
#SBATCH --partition=savio4_htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=preprocess_%j.log

module load anaconda3
source activate hubert_env

echo "=============================================="
echo "HuBERT Preprocessing"
echo "=============================================="
echo "Start time: $(date)"

# Verify environment
python -c "
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans
print('torch:', torch.__version__)
print('torchaudio:', torchaudio.__version__)
print('sklearn: OK')
try:
    from soundsig.sound import spectrogram
    print('soundsig: OK')
except:
    print('soundsig not available (needed for spectrogram features)')
"

echo ""
echo "Starting preprocessing..."

python /global/home/users/jonathanswang/pytorchAudio/examples/hubert/preprocess.py \
    --dataset short_zebra_finch \
    --root-dir /global/scratch/users/jonathanswang/dev \
    --feat-type spectrogram \
    --exp-dir /global/scratch/users/jonathanswang/temp_files/run2-6-26/ \
    --num-cluster 100 \
    --num-rank 5 \
    --layer-index 6 \
    --percent -1 \
    --kernel-size-ms 25 \
    --stride-ms 20 \
    --debug

echo ""
echo "End time: $(date)"
