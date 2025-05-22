#!/bin/bash
#
#SBATCH --partition=cpu_7cores
#SBATCH --qos=cpu_7cores
#SBATCH --job-name=gen
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python eye_analysis.py "gen" analysis_file_paths.txt duke+rider_analysis 
