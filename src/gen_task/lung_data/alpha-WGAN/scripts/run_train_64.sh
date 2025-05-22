#!/bin/bash
#
#SBATCH --partition=gpu_min32gb      # Reserved partition
#SBATCH --qos=gpu_min32gb         # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=alpha_wgan_200k_64            # Job name
#SBATCH -o train_64_200k.%N.%j.out           # File containing STDOUT output
#SBATCH -e train_64_200k.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 ../training_options/train_64.py --total_iter 200001 --checkpoint_dir "checkpoint_alpha_WGAN_64_200k" --samples_dir "samples_alpha_WGAN_64_200k" --losses_ext "alpha_WGAN_64_200k" 
