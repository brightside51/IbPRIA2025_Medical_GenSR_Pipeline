#!/bin/bash
#
#SBATCH --partition=gpu_min80GB
#SBATCH --qos=gpu_min80GB
#SBATCH --job-name=finepri
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python ~/pedropereira_msc23/Real-ESRGAN/realesrgan/train.py -opt finetune_realesrgan_x4plus_pairdata.yml --auto_resume