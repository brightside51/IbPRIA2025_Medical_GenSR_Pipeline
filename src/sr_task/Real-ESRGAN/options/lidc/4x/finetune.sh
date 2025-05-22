#!/bin/bash
#
#SBATCH --partition=debug_8gb
#SBATCH --qos=debug_8gb
#SBATCH --job-name=finedu
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

python ~/pedropereira_msc23/Real-ESRGAN/realesrgan/train.py -opt finetune_realesrgan_x4plus.yml --auto_resume
