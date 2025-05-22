#!/bin/bash
#
#SBATCH --partition=debug_8gb      # Reserved partition
#SBATCH --qos=debug_8gb          # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=cross_wgan            # Job name
#SBATCH -o slurm.%N.%j.out           # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 train.py
