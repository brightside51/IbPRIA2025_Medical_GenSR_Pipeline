#!/bin/bash
#
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=super_resolution
#SBATCH -o output.out
#SBATCH -e error.err

# --------------- VideoDiff V2 ---------------
#python model_generate_txt.py /nas-ctm01/homes/pfsousa/MetaBreast/logs/video_diffusion/V2/sample/ sample_videodiff_V2.txt /nas-ctm01/datasets/public/MEDICAL/super-res-models/duke/4x/finetune_RealESRGANx4plus_400k_pairdata/models/net_g_latest.pth ~pfsousa/super_resolution/Real-ESRGAN/experiments/videodiff_V2/

# -------------- MedDiff V1 --------------
#python model_generate_txt.py /nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/images_jpeg_lowres test_duke_rider_lidc1.txt ~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata/models/net_g_latest.pth ~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata/visualization
#python model_generate_txt.py /nas-ctm01/homes/pfsousa sample_meddiff_V1.txt ~pfsousa/super_resolution/Real-ESRGAN/experiments/pretrained_models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth ~pfsousa/super_resolution/Real-ESRGAN/experiments/meddiff_V1
#python model_generate_txt.py /nas-ctm01/homes/pfsousa/MedDiff/evaluation/V1/sampling_1 sample_meddiff_V1.txt /nas-ctm01/datasets/public/MEDICAL/super-res-models/lidc/final/finetune_RealESRGANx4plus_400k_pairdata/models/net_g_latest.pth ~pfsousa/super_resolution/Real-ESRGAN/experiments/meddiff_V1
#python model_generate_txt.py /nas-ctm01/homes/pfsousa/MedDiff/evaluation/V1/sampling_1 sample_meddiff_V1.txt /nas-ctm01/datasets/public/MEDICAL/super-res-models/rider/4x/finetune_RealESRGANx4plus_400k/models/net_g_latest.pth ~pfsousa/super_resolution/Real-ESRGAN/experiments/meddiff_V1
python model_generate_txt.py /nas-ctm01/homes/pfsousa/SuperResolution/super_resolution/Real-ESRGAN/experiments/swinir_lr/ swinir.txt /nas-ctm01/datasets/public/MEDICAL/super-res-models/rider/4x/finetune_RealESRGANx4plus_400k/models/net_g_latest.pth ~pfsousa/SuperResolution/super_resolution/Real-ESRGAN/experiments/


# ------------ Original Example ------------
#~pfsousa/super_resolution/Real-ESRGAN/experiments/pretrained_models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth
#~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata/models/net_g_latest.pth
#~/pedropereira_msc23/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_pairdata/visualization