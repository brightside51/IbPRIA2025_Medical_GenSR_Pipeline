#!/bin/bash
#
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb
#SBATCH --job-name=generate
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python temp.py /nas-ctm01/datasets/public/MEDICAL ~/pedropereira_msc23/test_lidc_duke_rider_blur_paired+.txt ~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata/models/net_g_latest.pth ~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata/visualization 4
