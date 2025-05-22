#!/bin/bash
#
#SBATCH --partition=cpu_7cores
#SBATCH --qos=cpu_7cores
#SBATCH --job-name=test
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python temp.py
