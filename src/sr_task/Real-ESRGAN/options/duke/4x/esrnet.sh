#!/bin/bash
#
#SBATCH --partition=gpu_min24GB
#SBATCH --qos=gpu_min24GB
#SBATCH --job-name=netdu4x
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python ~/pedropereira_msc23/Real-ESRGAN/realesrgan/train.py -opt train_realesrnet_x4plus.yml --auto_resume
