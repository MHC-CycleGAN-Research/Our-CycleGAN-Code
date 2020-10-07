#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

## Collections of all loss functions that we will use

def cycle_consistency_loss(real_images, generated_images):
    
    return tf.reduce_mean(tf.abs(real_images - generated_images))


def loss_generator(prob_fake_is_real):
    
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def loss_discriminator(prob_real_is_real, prob_fake_is_real):

    return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
            tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5


# def segmentation_loss(G, F, x, y):
    # Todo: Find the code of segmentor and implement as above

