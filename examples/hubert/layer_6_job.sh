#!/bin/bash
#SBATCH --job-name=hubert_l6_clusters
#SBATCH --account=fc_birdpow
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/users/keanumchandra/extract_clusters_%j.out
#SBATCH --error=/global/scratch/users/keanumchandra/extract_clusters_%j.err

# Activate Conda environment
module load anaconda3
source activate hubert_env

# Run cluster extraction script
/global/home/users/keanumchandra/.conda/envs/hubert_env/bin/python layer_6.py
