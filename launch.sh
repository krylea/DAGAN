#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --partition=t4v2
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=50GB
#SBATCH --exclude=gpu109

python3 train_omniglot_dagan.py --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title $1 --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5
