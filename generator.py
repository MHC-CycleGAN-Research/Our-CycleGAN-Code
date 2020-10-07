#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import ops
import utils

ngf = utils.ngf

## Neural Nets that generate images
class Generator:
    def __init__(name="generator", skip=False):
        self.name = name

    def __call__(self, inputgen):
        with tf.variable_scope(self.name):
            f = 7
            ks = 3
            padding = "REFLECT"

            pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [
                ks, ks], [0, 0]], padding)
            o_c1 = utils.general_conv2d(
                pad_input, ngf, f, f, 1, 1, 0.02, name="c1")
            o_c2 = utils.general_conv2d(
                o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
            o_c3 = utils.general_conv2d(
                o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

            o_r1 = utils.build_resnet_block(o_c3, ngf * 4, "r1", padding)
            o_r2 = utils.build_resnet_block(o_r1, ngf * 4, "r2", padding)
            o_r3 = utils.build_resnet_block(o_r2, ngf * 4, "r3", padding)
            o_r4 = utils.build_resnet_block(o_r3, ngf * 4, "r4", padding)
            o_r5 = utils.build_resnet_block(o_r4, ngf * 4, "r5", padding)
            o_r6 = utils.build_resnet_block(o_r5, ngf * 4, "r6", padding)
            o_r7 = utils.build_resnet_block(o_r6, ngf * 4, "r7", padding)
            o_r8 = utils.build_resnet_block(o_r7, ngf * 4, "r8", padding)
            o_r9 = utils.build_resnet_block(o_r8, ngf * 4, "r9", padding)

            o_c4 = utils.general_deconv2d(
                o_r9, [utils.BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
                "SAME", "c4")
            o_c5 = utils.general_deconv2d(
                o_c4, [utils.BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02,
                "SAME", "c5")
            o_c6 = utils.general_conv2d(o_c5, utils.IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c6",
                                     do_norm=False, do_relu=False)

            if skip is True:
                out_gen = tf.nn.tanh(inputgen + o_c6, "t1")
            else:
                out_gen = tf.nn.tanh(o_c6, "t1")

            return out_gen

