#!/bin/bash
#
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=evaluate
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python evaluate_model_txt.py "yes" test_only_duke+rider.txt /nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/images_jpeg ~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata/visualization ~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata
