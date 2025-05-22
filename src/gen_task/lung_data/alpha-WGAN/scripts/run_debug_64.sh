#!/bin/bash
#
#SBATCH --partition=debug_8gb      # Reserved partition
#SBATCH --qos=debug_8gb          # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=alpha_WGAN_debug_64            # Job name
#SBATCH -o slurm.%N.%j.out           # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 ../training_options/train_64.py --total_iter 200001 --checkpoint_dir "checkpoint_alpha_WGAN_64_200k" --samples_dir "samples_alpha_WGAN_64_200k" --losses_ext "alpha_WGAN_64_200k" 
