#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import utils
from utils import tf_count
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import functools

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models as Kmodels 
from tensorflow.python.keras import losses as Klosses
from tensorflow.python.keras import layers as Klayers
from tensorflow.python.keras import backend as K

## Collections of all loss functions that we will use
def cycle_consistency_loss(real_images, generated_images):
    
    return tf.reduce_mean(tf.abs(real_images - generated_images))

def generator_loss(prob_fake_is_real):
    
    return tf.reduce_mean(tf.math.squared_difference(prob_fake_is_real, 1))


def discriminator_loss(prob_real_is_real, prob_fake_is_real):

    return (tf.reduce_mean(tf.math.squared_difference(prob_real_is_real, 1)) +
            tf.reduce_mean(tf.math.squared_difference(prob_fake_is_real, 0))) * 0.5

def background_content_loss(real_images, generated_images, seg):

	seg_flipped = tf.math.squared_difference(seg, 1)
	seg_color = tf.image.grayscale_to_rgb(seg_flipped)
	img_diff = tf.math.squared_difference(real_images, generated_images)
	background_diff = tf.math.multiply(img_diff, seg_color)

	return tf.reduce_sum(background_diff) / tf.reduce_sum(seg_color)


def get_VGGmodel():
    """ Creates our model with access to intermediate layers. 
  
    This function will load the VGG19 model and access the intermediate layers. 
    These layers will then be used to create a new model that will take input image
        and return the outputs from these intermediate layers from the VGG model. 
  
    Returns:
        returns a keras model that takes image inputs and outputs the style and 
            content intermediate layers. 
    """
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Get output layers corresponding to style and content layers 
    style_outputs = [vgg.get_layer(name).output for name in utils.STYLE_LAYERS]
    content_outputs = [vgg.get_layer(name).output for name in utils.CONTENT_LAYERS]
    model_outputs = style_outputs + content_outputs

    # Build model 
    return Kmodels.Model(vgg.input, model_outputs)


def foreground_style_loss(generated_image, real_image, seg_gen, seg_real):
    
    # load VGG model and disable trainability
    VGGmodel = get_VGGmodel() 
    for layer in VGGmodel.layers:
        layer.trainable = False
        
    # get VGGmodel outputs for images
    # Feed our images through our model. This will give us the content and 
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    real_output = VGGmodel(real_image)
    gen_output = VGGmodel(generated_image)
    
    # get VGGmodel outputs for segmentation masks
    real_color_mask = tf.image.grayscale_to_rgb(seg_real)
    gen_color_mask = tf.image.grayscale_to_rgb(seg_gen)
    real_output_mask = VGGmodel(real_color_mask)
    gen_output_mask = VGGmodel(gen_color_mask)

    # extract style features from layer output
    real_style_features = [style_layer[0] for style_layer in real_output[:utils.N_STYLE_LAYERS]]
    gen_style_features = [style_layer[0] for style_layer in gen_output[:utils.N_STYLE_LAYERS]]

    # apply mask to the layer outputs
    real_style_features_mask = [style_layer[0] for style_layer in real_output_mask[:utils.N_STYLE_LAYERS]]
    gen_style_features_mask = [style_layer[0] for style_layer in gen_output_mask[:utils.N_STYLE_LAYERS]]
    masked_real_style_features = []
    masked_gen_style_features = []
    for i in range(len(real_style_features)):
        masked_real_style_features.append(tf.math.multiply(real_style_features[i],real_style_features_mask[i]))
        masked_gen_style_features.append(tf.math.multiply(gen_style_features[i],gen_style_features_mask[i]))

    # calculate gram matrix response of the masked features
    gram_real_style_features = [utils.gram_matrix(masked_style_feature) for masked_style_feature in masked_real_style_features]
    gram_gen_style_features = [utils.gram_matrix(masked_style_feature) for masked_style_feature in masked_gen_style_features]
    
    # calculate style loss
    style_score = 0

    # (1) we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(utils.N_STYLE_LAYERS)

    # (2) accumulate style losses from all layers
    for gram_target, gram_base, img_base in zip(gram_real_style_features, gram_gen_style_features, masked_gen_style_features):

        # (3) Expects three images of dimension h, w, c ==  height, width, num of filters
        height, width, channels = img_base.get_shape().as_list()
        
        # (4) scale the loss ***at a given layer*** by the size of the feature map and the number of filters
        #     element-wise matrix multiplication for each layer
        layer_style_loss = tf.reduce_mean(tf.square(gram_base - gram_target)) / (4. * (channels ** 2) * (width * height) ** 2)
        style_score += weight_per_style_layer * layer_style_loss

    style_score *= utils.STYLE_WEIGHT

    return style_score

