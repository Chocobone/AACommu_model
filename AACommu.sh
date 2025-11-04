#!/usr/bin/bash

#SBATCH -J AACommu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g8
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out


/data/yho7374/anaconda3/bin/conda init
source ~/.bashrc
conda activate training

python thesis_aac_project.py

exit 0
