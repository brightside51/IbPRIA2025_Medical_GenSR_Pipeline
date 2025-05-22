#!/bin/bash
#
#SBATCH --partition=gpu_min11GB      # Reserved partition
#SBATCH --qos=gpu_min11gb          # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=pixel_analysis            # Job name
#SBATCH -o slurm.%N.%j.out           # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 -u pixel_analysis.py
