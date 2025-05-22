#!/bin/bash
#
#SBATCH --partition=gpu_min11GB      # Reserved partition
#SBATCH --qos=gpu_min11gb          # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=mask_analysis            # Job name
#SBATCH -o slurm_128.%N.%j.out           # File containing STDOUT output
#SBATCH -e slurm_128.%N.%j.err           # File containing STDERR output


echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
# (...)
# python3 -u dicom_to_npy_conversion.py > prints.out
python3 -u mask_analysis.py > prints.out