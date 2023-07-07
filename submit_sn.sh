#!/bin/bash
#
#SBATCH --job-name=kgc
#SBATCH --output=ft-dolly-v2-alpaca.txt
#
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH -p gpu
#SBATCH -t 5-00:00:00

srun hostname
source activate kgc
python fine_tuning_dolly_2_0_with_lora_and_alpaca.py
