#!/bin/bash
#
#SBATCH --partition=gpu_min80GB      # Reserved partition
#SBATCH --qos=gpu_min80gb          # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=3DPGGAN           # Job name
#SBATCH -o train_64.%N.%j.out           # File containing STDOUT output
#SBATCH -e train_64.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
# python3 train_options.py --total_iter 200001 --checkpoint_dir "checkpoint_o_200k_128" --samples_dir "samples_128_200k" --losses_ext "_128_200k" --slices_size 128
python3 train3D.py