#!/usr/bin/env python
# coding: utf-8

# In[3]:

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import utils

ndf = utils.ndf

## Neural Net that provide probability that the input is real or not
class Discriminator:
    def __init__(self, name="discriminator"):
        self.name = name
    
    def __call__(self, inputdisc):
        with tf.variable_scope(self.name):
            f = 4

            o_c1 = utils.general_conv2d(inputdisc, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm=False,
                                     relufactor=0.2)
            o_c2 = utils.general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
            o_c3 = utils.general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
            o_c4 = utils.general_conv2d(o_c3, ndf * 8, f, f, 1, 1,
                                     0.02, "SAME", "c4", relufactor=0.2)
            o_c5 = utils.general_conv2d( o_c4, 1, f, f, 1, 1, 
                                     0.02, "SAME", "c5", do_norm=False, do_relu=False)

            return o_c5
    


# In[ ]:




