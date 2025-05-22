#!/bin/bash
#
#SBATCH --partition=gpu_min12gb      # Reserved partition
#SBATCH --qos=gpu_min12gb       # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=hagan_fid       # Job name
#SBATCH -o fid_cross_128_200.%N.%j.out           # File containing STDOUT output
#SBATCH -e fid_cross_128_200.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 fid_score.py 