#!/bin/bash
#Set job requirements
#SBATCH --job-name=kgc
#SBATCH --output=ft-dolly2.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=defq
#SBATCH --time=5-00:00:00

module load cuda11.7/toolkit
srun python3 fine_tuninig_dolly_2_0_with_lora_and_alpaca.py
