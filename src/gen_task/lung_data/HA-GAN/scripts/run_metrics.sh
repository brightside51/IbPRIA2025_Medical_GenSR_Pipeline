#!/bin/bash
#
#SBATCH --partition=gpu_min24gb      # Reserved partition
#SBATCH --qos=gpu_min24gb         # QoS level. Must match the partition name. External users must add the suffix "_ext".

#SBATCH --job-name=HA-GAN_256_un       # Job name
#SBATCH -o metrics_HA-GAN_256_un.%N.%j.out           # File containing STDOUT output
#SBATCH -e metrics_HA-GAN_256_un.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 ../metrics/fid_score.py
