#!/bin/bash
#
#SBATCH --partition=debug_8GB      # Reserved partition
#SBATCH --qos=debug_8GB          # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=PGGAN_debug            # Job name
#SBATCH -o slurm.%N.%j.out           # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
# python3 train_options.py --total_iter 200001 --checkpoint_dir "checkpoint_o_200k_128" --samples_dir "samples_128_200k" --losses_ext "_128_200k" --slices_size 128
python3 train3D.py