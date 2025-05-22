# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.

data_dir = 'datasets'
result_dir = 'results_20'

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']          = True     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
#env.CUDA_VISIBLE_DEVICES                       =  '0' #'0,1,2,3'  Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '1'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

train       = EasyDict(func='train3D.train_progressive_gan')  # Options for main training func.

#----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc        = 'pgan'                                           # Description string included in result subdir name.
random_seed = 1000                                             # Global random seed.
dataset     = EasyDict()                                       # Options for dataset.load_dataset().
G           = EasyDict(func='networks3D.G_paper')             # Options for generator network.
D           = EasyDict(func='networks3D.D_paper')             # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)    # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)    # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss3D.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss3D.D_wgangp_acgan')          # Options for discriminator loss.
D_accuracy  = EasyDict(func='loss3D.D_wgangp_acgan_accuracy') # Options for discriminator accuracy
sched       = EasyDict()                                       # Options for train.TrainingSchedule.
grid        = EasyDict(size='1080p', layout='random')          # Options for train.setup_snapshot_image_grid().

# Dataset (choose one)
#desc += '-LIDC256';            dataset = EasyDict(tfrecord_dir='/gdrive/MyDrive/Data/LIDC/TFRecords_5') ; 
desc += '-LIDC256';            dataset = EasyDict(tfrecord_dir='/home/tpereira/artur/TFRecords_5') ; 
desc += '-fp32' 

#train = EasyDict(func='util_scripts.generate_fake_images', run_id=23, num_pngs=1000); num_gpus = 1; desc = 'fake-images-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_fake_images', run_id=23, grid_size=[15,8], num_pngs=10, image_shrink=4); num_gpus = 1; desc = 'fake-grids-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_interpolation_video', run_id=39, grid_size=[1,4], duration_sec=0.5, smoothing_sec=1.0, minibatch_size=2); num_gpus = 2; desc = 'interpolation-video-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_training_video', run_id=23, duration_sec=20.0); num_gpus = 1; desc = 'training-video-' + str(train.run_id)

#train = EasyDict(func='util_scripts.generate_fake_images', run_id='results_20/007-pgan-LIDC256-preset-v2-2gpu-fp32-GRAPH-HIST/network-snapshot-002602.pkl', snapshot=2602, num_pngs=100, minibatch_size=2); num_gpus = 2; desc = 'fake-images-' + str(train.run_id)
train = EasyDict(func='util_scripts.generate_fake_images', run_id=39, snapshot=3805, num_pngs=1000, minibatch_size=2); num_gpus = 2; desc = 'fake-images-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=4, log='metric-msssim-2.txt', metrics=['msssim'], num_images=100, real_passes=1); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)

#----------------------------------------------------------------------------
