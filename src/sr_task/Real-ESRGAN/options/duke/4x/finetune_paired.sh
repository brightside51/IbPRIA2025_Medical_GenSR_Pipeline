#!/bin/bash
#
#SBATCH --partition=gpu_min24GB
#SBATCH --qos=gpu_min24GB
#SBATCH --job-name=finepdu
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python ~/pedropereira_msc23/Real-ESRGAN/realesrgan/train.py -opt finetune_realesrgan_x4plus_pairdata.yml --auto_resume
