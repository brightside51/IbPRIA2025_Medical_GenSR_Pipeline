#!/bin/bash
#
#SBATCH --partition=gpu_min80GB
#SBATCH --qos=gpu_min80GB
#SBATCH --job-name=netri2x
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python ~/pedropereira_msc23/Real-ESRGAN/realesrgan/train.py -opt train_realesrnet_x2plus.yml --auto_resume