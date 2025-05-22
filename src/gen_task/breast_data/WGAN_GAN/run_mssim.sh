#!/bin/bash
#
#SBATCH --partition=gpu_min80GB      # Reserved partition
#SBATCH --qos=gpu_min80gb        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=exp_WGAN       # Job name
#SBATCH -o test_mssim.%N.%j.out           # File containing STDOUT output
#SBATCH -e test_mssim.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 mssim.py 

