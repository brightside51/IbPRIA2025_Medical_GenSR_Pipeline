#!/bin/bash
#
#SBATCH --partition=gpu_min32gb      # Reserved partition
#SBATCH --qos=gpu_min32gb          # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=130_4_mssim_real_64            # Job name
#SBATCH -o 130_4_mssim_real_64.%N.%j.out           # File containing STDOUT output
#SBATCH -e 130_4_mssim_real_64.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 ../metrics/min_max_mssim.py
