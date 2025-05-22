#!/bin/bash
#
#SBATCH --partition=gpu_min11GB     # Reserved partition
#SBATCH --qos=gpu_min11gb         # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=meddifus       # Job name
#SBATCH -o train_vqgan.%N.%j.out           # File containing STDOUT output
#SBATCH -e train_vqgan.%N.%j.err           # File containing STDERR output

echo "Running job in reserved partition"
export PYTHONPATH=$PWD
# Commands / scripts to run (e.g., python3 train.py)
# (...)
 #PL_TORCH_DISTRIBUTED_BACKEND=gloo CUDA_VISIBLE_DEVICES=1 python3 train/train_vqgan.py --gpus 1 --default_root_dir "/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1" --precision 16 --embedding_dim 8 --n_hiddens 16 --downsample 8 8 8 --num_workers 32 --gradient_clip_val 1.0 --lr 3e-4 --discriminator_iter_start 10000 --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1 --gan_feat_weight 4 --batch_size 2 --n_codes 16384 --accumulate_grad_batches 1 
 PL_TORCH_DISTRIBUTED_BACKEND=gloo python3 train/train_vqgan.py dataset=default dataset.root_dir=/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1 model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='own_dataset' model.precision=16 model.embedding_dim=8 model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384 model.accumulate_grad_batches=1 