#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --account=fc_birdpow
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00

module load anaconda3
source activate hubert_env

/global/home/users/keanumchandra/.conda/envs/hubert_env/bin/python train.py \
  --dataset-path /global/scratch/users/keanumchandra/split_audio_2 \
  --feature-type hubert \
  --num-classes 100 \
  --gpus 1 \
  --max-updates 1000

