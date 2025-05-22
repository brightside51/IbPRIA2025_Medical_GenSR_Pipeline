#!/bin/bash
#
#SBATCH --partition=cpu_7cores
#SBATCH --qos=cpu_7cores
#SBATCH --job-name=gen
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python scripts/generate_meta_info_pairdata.py --input /nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/images_jpeg /nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/images_jpeg_lowres --meta_info ~/pedropereira_msc23/full_lidc_paired+.txt
