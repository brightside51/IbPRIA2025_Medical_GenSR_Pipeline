#!/bin/bash
#
#SBATCH --partition=gpu_min24gb      # Reserved partition
#SBATCH --qos=gpu_min24gb          # QoS level. Must match the partition name. External users must add the suffix "_ext".

#SBATCH --job-name=fid_110k_WGAN_64       # Job name
#SBATCH -o fid_110k_64.%N.%j.out           # File containing STDOUT output
#SBATCH -e fid_110k_64.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 ../metrics/fid_score.py
