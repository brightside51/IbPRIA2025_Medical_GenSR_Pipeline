#!/bin/bash
#
#SBATCH --partition=debug_8gb      # Reserved partition
#SBATCH --qos=debug_8gb          # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=HA-GAN_256_un           # Job name
#SBATCH -o HA-GAN_256_un.%N.%j.out           # File containing STDOUT output
#SBATCH -e HA-GAN_256_un.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python ../train.py --workers 8 --img-size 256 --num-class 0 --exp-name 'HA_GAN_run1_256' --data-dir DATA_DIR