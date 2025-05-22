#!/bin/bash
#
#SBATCH --partition=debug_8gb     # Reserved partition
#SBATCH --qos=debug_8gb       # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=debug_hagan       # Job name
#SBATCH -o debug_hagan_128.%N.%j.out           # File containing STDOUT output
#SBATCH -e debug_hagan_128.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 train_2.py --config config.json
