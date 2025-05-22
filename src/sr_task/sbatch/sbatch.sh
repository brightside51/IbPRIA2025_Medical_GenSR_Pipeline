#!/bin/bash
#
#SBATCH --partition=cpu_7cores
#SBATCH --qos=cpu_7cores
#SBATCH --job-name=process
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python crop.py same_structure_crop.txt /nas-ctm01/datasets/public/MEDICAL/super-res-models/duke+rider/4x/train_RealESRGANx4plus_400k_B12G4/visualization ~/duke+rider/from_scratch_crop
