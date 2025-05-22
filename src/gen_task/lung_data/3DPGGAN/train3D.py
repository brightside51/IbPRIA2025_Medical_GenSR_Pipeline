# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
import sys
import config3D
import tfutil3D
import dataset3D
import misc3D
import loss3D

#----------------------------------------------------------------------------
# Choose the size and contents of the image snapshot grids that are exported
# periodically during training.

def setup_snapshot_image_grid(G, training_set,
    size    = '1080p',      # '1080p' = to be viewed on 1080p display, '4k' = to be viewed on 4k display.
    layout  = 'random'):    # 'random' = grid contents are selected randomly, 'row_per_class' = each row corresponds to one class label.

    # Select size.
    gw = 1; gh = 1
    if size == '1080p':
        gw = np.clip(1920 // G.output_shape[3], 3, 32)
        gh = np.clip(1080 // G.output_shape[2], 2, 32)
    if size == '4k':
        gw = np.clip(3840 // G.output_shape[3], 7, 32)
        gh = np.clip(2160 // G.output_shape[2], 4, 32)
    gw = 1; gh = 5 # number examples every snapshot

    # Fill in reals and labels.
    reals = np.zeros([gw * gh] + training_set.shape, dtype=training_set.dtype)
    labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        while True:
            real, label = training_set.get_minibatch_np(1)
            if layout == 'row_per_class' and training_set.label_size > 0:
                if label[0, y % training_set.label_size] == 0.0:
                    continue
            reals[idx] = real[0]
            labels[idx] = label[0]
            break

    # Generate latents.
    latents = misc3D.random_latents(gw * gh, G)
    return (gw, gh), reals, labels, latents

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            #x = tf.cast(x, tf.float16)
            x = misc3D.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3], s[4]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[4]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2, s[4]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5, 7], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3], s[4]])
            x = tfutil3D.lerp(x, y, lod - tf.floor(lod)) 
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            #factor = (2 ** np.floor(lod.data)).astype(int32)
            #x = tf.compat.v2.keras.layers.UpSampling3D(size=(factor,factor,factor),data_format='channels_first')(x)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1, s[4], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor, s[4] * factor])
        return x

#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_initial_resolution  = 4,        # Image resolution used at the beginning.
        lod_training_kimg       = 200,  # 1000 600 6000     # Thousands of real images to show before doubling the resolution.
        lod_transition_kimg     = 200,  # 1000 600 6000    # Thousands of real images to show when fading in new layers.
        minibatch_base          = 4,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
        G_lrate_base            = 0.001, # 0.0003    # Learning rate for the generator.
        G_lrate_dict            = {},       # Resolution-specific overrides.
        D_lrate_base            = 0.001, # 0.0003,    # Learning rate for the discriminator.
        D_lrate_dict            = {},       # Resolution-specific overrides.
        tick_kimg_base          = 160,      # Default interval of progress snapshots.
        tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:20, 1024:10}): # Resolution-specific overrides.

        # Training phase.
        self.kimg = cur_nimg / 1000.0
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = self.kimg - phase_idx * phase_dur

        # Level-of-detail and resolution.
        self.lod = training_set.resolution_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(self.lod)))

        # Minibatch size.
        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % config3D.num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * config3D.num_gpus)

        # Other parameters.
        self.G_lrate = G_lrate_dict.get(self.resolution, G_lrate_base)
        self.D_lrate = D_lrate_dict.get(self.resolution, D_lrate_base)
        self.tick_kimg = tick_kimg_dict.get(self.resolution, tick_kimg_base)

#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_progressive_gan(
    G_smoothing             = 0.999,        # Exponential running average of generator weights.
    D_repeats               = 1,            # How many times the discriminator is trained per update
    G_repeats               = 1,            # How many times the generator is trained per update
    minibatch_repeats       = 4,            # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,         # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 12000,        # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,        # Enable mirror augment?
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    numpy_snapshot_ticks    = 1,            # How often to export numpy snapshots?
    image_snapshot_ticks    = 1,            # How often to export image snapshots?
    network_snapshot_ticks  = 1,            # How often to export network snapshots?
    save_tf_graph           = False,        # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,        # Include weight histograms in the tfevents file?
    resume_run_id           = None,         # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,         # Snapshot index to resume training from, None = autodetect.
    resume_kimg             = 0.0,          # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0):         # Assumed wallclock time at the beginning. Affects reporting.

    maintenance_start_time = time.time()
    training_set = dataset3D.load_dataset(data_dir=config3D.data_dir, verbose=True, **config3D.dataset)
    
    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc3D.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            G, D, Gs = misc3D.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            G = tfutil3D.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config3D.G)
            D = tfutil3D.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config3D.D)
            Gs = G.clone('Gs')
            #if resume_run_id != 1245:
            #    network_pkl = misc3D.locate_network_pkl(resume_run_id, resume_snapshot)
            #    print('Loading networks from "%s"...' % network_pkl)
            #    G1, D1, G1s = misc3D.load_pkl(network_pkl)
            #    G.copy_trainables_from(G1)
            #    D.copy_trainables_from(D1)
                #G.__setstate__(G1s.__getstate__())

            
        Gs_update_op = Gs.setup_as_moving_average_of(G, beta=G_smoothing)
    G.print_layers(); D.print_layers(); 

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // config3D.num_gpus
        reals, labels   = training_set.get_minibatch_tf()
        reals_split     = tf.split(reals, config3D.num_gpus)
        labels_split    = tf.split(labels, config3D.num_gpus)
    G_opt = tfutil3D.Optimizer(name='TrainG', learning_rate=lrate_in, **config3D.G_opt)
    D_opt = tfutil3D.Optimizer(name='TrainD', learning_rate=lrate_in, **config3D.D_opt)
    for gpu in range(config3D.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in)]
            
            reals_gpu = process_reals(reals_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            labels_gpu = labels_split[gpu]
            with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
                G_loss = tfutil3D.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set, minibatch_size=minibatch_split, **config3D.G_loss)
                #print("G loss:", G_loss.eval())
            with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
                D_loss = tfutil3D.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals_gpu, labels=labels_gpu, **config3D.D_loss)
                #print("D loss:", D_loss.eval())
            with tf.name_scope('D_accuracy'), tf.control_dependencies(lod_assign_ops):
                D_accuracy = tfutil3D.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals_gpu, labels=labels_gpu, **config3D.D_accuracy)    
                #print("D acc:", D_accuracy.eval())
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()

    print('Setting up snapshot image grid...', end='')
    grid_size, grid_reals, grid_labels, grid_latents = setup_snapshot_image_grid(G, training_set, **config3D.grid)
    #print('!!!Latents shape!!!!', grid_latents.shape)
    sched = TrainingSchedule(total_kimg * 1000, training_set, **config3D.sched)
    #print('ok')
    
    #print("first run ")
    #print("train", sched.minibatch)
    grid_fakes = Gs.run(grid_latents, grid_labels, minibatch_size=sched.minibatch//config3D.num_gpus)
    #print('ok')

    print('Setting up result dir...', end='')
    result_subdir = misc3D.create_result_subdir(config3D.result_dir, config3D.desc)
    #misc3D.save_image_grid(grid_reals, os.path.join(result_subdir, 'reals.png'), os.path.join(result_subdir, 'reals.npy'),  drange=training_set.dynamic_range, grid_size=grid_size)
    #if resume_run_id is None: # non trained wheights
    grid_fakes = Gs.run(grid_latents, grid_labels, minibatch_size=sched.minibatch//config3D.num_gpus)
    misc3D.save_image_grid(grid_fakes, os.path.join(result_subdir, 'Gs_fakes%06d.png' % (resume_kimg // 1000)), drange=drange_net, grid_size=grid_size)

    #grid_fakes = G.run(grid_latents, grid_labels, minibatch_size=sched.minibatch//config3D.num_gpus)
    #misc3D.save_image_grid(grid_fakes, os.path.join(result_subdir, 'G_fakes%06d.png' % (resume_kimg // 1000)), drange=drange_net, grid_size=grid_size)
    #misc3D.save_image_grid(grid_fakes, os.path.join(result_subdir, 'fakes%06d.png' % 0), os.path.join(result_subdir, 'fakes%06d.npy' % 0), drange=drange_net, grid_size=grid_size)

    summary_log = tf.summary.FileWriter(result_subdir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()

    print('Training...')
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time - resume_time
    prev_lod = -1.0
    while cur_nimg < total_kimg * 1000:
        #print(cur_nimg, '/', total_kimg * 1000, round(cur_nimg / (total_kimg * 1000)  * 100, 2), end='\n')
        # Choose training parameters and configure training ops.
        sched = TrainingSchedule(cur_nimg, training_set, **config3D.sched)
        training_set.configure(sched.minibatch, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        for repeat in range(minibatch_repeats):
            #print('Train loop: lod', sched.lod, 'lrate', sched.D_lrate, 'minibatch', sched.minibatch)
            for _ in range(D_repeats):
                tfutil3D.run([D_train_op, Gs_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
                cur_nimg += sched.minibatch
            for _ in range(G_repeats):    
                tfutil3D.run([G_train_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})

       
            
        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 300 or done:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 300.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            total_time = cur_time - train_start_time
            maintenance_time = tick_start_time - maintenance_start_time
            maintenance_start_time = cur_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d  lrate %-5.4f  time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %.1f' % (
                tfutil3D.autosummary('Progress/tick', cur_tick),
                tfutil3D.autosummary('Progress/kimg', cur_nimg / 1000.0),
                tfutil3D.autosummary('Progress/lod', sched.lod),
                tfutil3D.autosummary('Progress/minibatch', sched.minibatch),
                tfutil3D.autosummary('Progress/Glrate', sched.G_lrate),
                misc3D.format_time(tfutil3D.autosummary('Timing/total_sec', total_time)),
                tfutil3D.autosummary('Timing/sec_per_tick', tick_time),
                tfutil3D.autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                tfutil3D.autosummary('Timing/maintenance_sec', maintenance_time))),
            tfutil3D.autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            tfutil3D.autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            tfutil3D.save_summaries(summary_log, cur_nimg)
            

            print('\r',cur_nimg, '/', total_kimg * 1000, round(cur_nimg / (total_kimg * 1000)  * 100, 2), end='\n\n')
            # Save snapshots.
            if cur_tick % image_snapshot_ticks == 0 or done:
                # Gs
                grid_fakes = Gs.run(grid_latents, grid_labels, minibatch_size=sched.minibatch//config3D.num_gpus)
                misc3D.save_image_grid(grid_fakes, os.path.join(result_subdir, 'Gs_fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
                # G
                #grid_fakes_1 = G.run(grid_latents, grid_labels, minibatch_size=sched.minibatch//config3D.num_gpus)
                #misc3D.save_image_grid(grid_fakes_1, os.path.join(result_subdir, 'G_fakes%06d_G.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
                if cur_tick % numpy_snapshot_ticks == 0 or done:
                    misc3D.save_numpy(grid_fakes, os.path.join(result_subdir, 'fakes%06d.npy' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
            if cur_tick % network_snapshot_ticks == 0 or done:
                misc3D.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000)))

            # Record start time of the next tick.
            tick_start_time = time.time()

    # Write final results.
    misc3D.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-final.pkl'))
    summary_log.close()
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    misc3D.init_output_logging()
    np.random.seed(config3D.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config3D.env)
    tfutil3D.init_tf(config3D.tf_config)
    print('Running %s()...' % config3D.train['func'])
    tfutil3D.call_func_by_name(**config3D.train)
    print('Exiting...')

#----------------------------------------------------------------------------
