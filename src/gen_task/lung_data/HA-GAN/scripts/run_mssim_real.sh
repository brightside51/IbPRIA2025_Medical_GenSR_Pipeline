#!/bin/bash
#
#SBATCH --partition=debug_8gb      # Reserved partition
#SBATCH --qos=debug_8gb          # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=80_mssim_real_64            # Job name
#SBATCH -o 80_mssim_real_64.%N.%j.out           # File containing STDOUT output
#SBATCH -e 80_mssim_real_64.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 ../metrics/mssim_real.py
