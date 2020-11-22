#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from utils import tf_count
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf


## Collections of all loss functions that we will use
def cycle_consistency_loss(real_images, generated_images):
    
    return tf.reduce_mean(tf.abs(real_images - generated_images))

def generator_loss(prob_fake_is_real):
    
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def discriminator_loss(prob_real_is_real, prob_fake_is_real):

    return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
            tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5

def background_content_loss(real_images, generated_images, seg):

	seg_flipped = tf.squared_difference(seg, 1)
	seg_color = tf.image.grayscale_to_rgb(seg_flipped)
	img_diff = tf.squared_difference(real_images, generated_images)
	background_diff = tf.math.multiply(img_diff, seg_color)

	return tf.reduce_sum(background_diff) / tf.reduce_sum(seg_color)


def foreground_style_loss(generated_image, real_image, seg_gen, seg_real):

	seg_gen_color = tf.image.grayscale_to_rgb(seg_gen)
	seg_real_color = tf.image.grayscale_to_rgb(seg_real)
	# TODO: implement the style loss

	return tf.zeros_like(tf.reduce_sum(seg_real))

