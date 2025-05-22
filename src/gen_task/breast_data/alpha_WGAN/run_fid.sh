#!/bin/bash
#
#SBATCH --partition=gpu_min32gb      # Reserved partition
#SBATCH --qos=gpu_min32gb       # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=fid_awgan       # Job name
#SBATCH -o fid_cross_128.%N.%j.out           # File containing STDOUT output
#SBATCH -e fid_cross_128.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 fid_score.py