# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()

import tfutil3D

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
    cond_weight = 1.0): # Weight of the conditioning term.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight

    loss = tfutil3D.autosummary('Loss/G_loss', loss)
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    #print(np.float32(real_scores_out))
    #print(np.float32(real_labels_out))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil3D.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil3D.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    raw_loss = loss
    raw_loss = tfutil3D.autosummary('Loss/D_raw_loss', raw_loss)
    
    real_accuracy = tf.math.reduce_sum(real_labels_out)
    real_accuracy = tfutil3D.autosummary('Loss/D_real_accuracy', real_accuracy)

    fake_accuracy = tf.math.divide(tf.math.subtract(fp32(minibatch_size), tf.math.reduce_sum(fake_labels_out)), fp32(minibatch_size))
    fake_accuracy = tfutil3D.autosummary('Loss/D_fake_accuracy', fake_accuracy)

    half = tf.constant(0.5)
    accuracy = tf.math.add(tf.math.multiply(real_accuracy,half) , tf.math.multiply(fake_accuracy,half) )
    accuracy = tfutil3D.autosummary('Loss/D_accuracy', accuracy)

    
    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil3D.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil3D.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3,4]))
        mixed_norms = tfutil3D.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil3D.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil3D.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil3D.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight

    loss = tfutil3D.autosummary('Loss/D_loss', loss)
    return loss


#----------------------------------------------------------------------------
# Discriminator accuracy function 

def D_wgangp_acgan_accuracy(G, D, opt, training_set, minibatch_size, reals, labels):

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=False)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=False))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=False))

    #accuracy2 = 0.5 * np.sum(real_labels_out) / fp32(minibatch_size) + 0.5 * (fp32(minibatch_size) - fp32(fake_labels_out)) / fp32(minibatch_size)
    accuracy2 = 0.5 * fp32(real_labels_out) / fp32(minibatch_size) + 0.5 * (fp32(minibatch_size) - fp32(fake_labels_out)) / fp32(minibatch_size)
    accuracy = tfutil3D.autosummary('Loss/D_accuracy_2', accuracy2)

    return accuracy2

#----------------------------------------------------------------------------
