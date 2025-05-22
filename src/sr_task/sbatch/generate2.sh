#!/bin/bash
#
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb
#SBATCH --job-name=generate
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python model_generate_txt.py /nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/images_jpeg_lowres test_duke_rider_lidc1.txt ~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata/models/net_g_latest.pth ~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata/visualization
