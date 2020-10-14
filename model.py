#!/usr/bin/env python
# coding: utf-8

# In[5]:



import tensorflow as tf
import json
import numpy as np
import os
import random
import click
from scipy.misc import imsave
import utils
import cyclegan_datasets
import discriminator
import generator
import data_loader, loss
from datetime import datetime

class CycleGAN:
    def __init__(self, 
                 pool_size,
                 lambda1, # sugegsted value: 10
                 lambda2, # suggested value: 10
                 output_root_dir,
                 to_restore,
                 _base_lr,
                 max_step, 
                 dataset_name, 
                 checkpoint_dir, 
                 do_flipping, 
                 skip
                ):
        
        self._pool_size = pool_size
        self._size_before_crop = 286

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        
        self._base_lr = _base_lr
        self._max_step = max_step
        
        self._num_imgs_to_save = 20
        self._to_restore = to_restore
        
        self._dataset_name = dataset_name
        self._checkpoint_dir = checkpoint_dir
        self._do_flipping = do_flipping
        self._skip = skip
    
        ## Define the hyperparameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.beta1 = 0.5
        
        self.fake_images_X = np.zeros(
            (self._pool_size, 1, utils.IMG_HEIGHT, utils.IMG_WIDTH,
             utils.IMG_CHANNELS)
        )
        self.fake_images_Y = np.zeros(
            (self._pool_size, 1, utils.IMG_HEIGHT, utils.IMG_WIDTH,
             utils.IMG_CHANNELS)
        )

        
    def model(self):
        self.input_x = tf.placeholder(
            tf.float32, shape=[
                1,
                utils.IMG_HEIGHT,
                utils.IMG_WIDTH,
                utils.IMG_CHANNELS
            ], name="input_X")
        self.input_y = tf.placeholder(
            tf.float32, shape=[
                1,
                utils.IMG_HEIGHT,
                utils.IMG_WIDTH,
                utils.IMG_CHANNELS
            ], name="input_Y")
                       
        ## Define a placeholder fed with fake x/fake y
        self.fake_pool_X = tf.placeholder(
            tf.float32, shape=[
                None,
                utils.IMG_HEIGHT,
                utils.IMG_WIDTH,
                utils.IMG_CHANNELS
            ], name="fake_pool_X")
        
        self.fake_pool_Y = tf.placeholder(
            tf.float32, shape=[
                None,
                utils.IMG_HEIGHT,
                utils.IMG_WIDTH,
                utils.IMG_CHANNELS
            ], name="fake_pool_Y")
        
        self.global_step = tf.Variable(1, name="global_step")

        self.num_fake_inputs = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

        self.images_x = self.input_x
        self.images_y = self.input_y
        
        self.G = generator.Generator("G_X", skip=self._skip)
        self.F = generator.Generator("G_Y", skip=self._skip)
        self.D_X = discriminator.Discriminator('D_X')
        self.D_Y = discriminator.Discriminator('D_Y')
        
        with tf.variable_scope("Model") as scope:

            self.prob_real_x_is_real = self.D_X(self.images_x)
            self.prob_real_y_is_real = self.D_Y(self.images_y)
        
            self.fake_images_y = self.G(self.images_x)
            self.fake_images_x = self.F(self.images_y)
            
            scope.reuse_variables()
            
            self.prob_fake_x_is_real = self.D_X(self.fake_images_x)
            self.prob_fake_y_is_real = self.D_Y(self.fake_images_y)

            self.cycle_images_x = self.G(self.fake_images_y)
            self.cycle_images_y = self.F(self.fake_images_x)
            
            scope.reuse_variables()

            self.prob_fake_pool_x_is_real = self.D_X(self.fake_pool_X)
            self.prob_fake_pool_y_is_real = self.D_Y(self.fake_pool_Y)
        
        
    def compute_losses(self):    
        
        G_cycle_loss = self.lambda1 * loss.cycle_consistency_loss(self.input_x, self.cycle_images_x)
        F_cycle_loss = self.lambda2 * loss.cycle_consistency_loss(self.input_y, self.cycle_images_y)
        
        G_gan_loss = loss.loss_generator(self.prob_fake_x_is_real)
        F_gan_loss = loss.loss_generator(self.prob_fake_y_is_real)
        
        G_loss = G_cycle_loss + F_cycle_loss + F_gan_loss
        F_loss = F_cycle_loss + G_cycle_loss + G_gan_loss
            
        D_X_loss = loss.loss_discriminator(self.prob_real_x_is_real, self.prob_fake_pool_x_is_real)
        D_Y_loss = loss.loss_discriminator(self.prob_real_y_is_real, self.prob_fake_pool_y_is_real)
        
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
        
        self.model_vars = tf.trainable_variables()
        
        d_X_vars = [var for var in self.model_vars if 'D_X' in var.name]
        g_X_vars = [var for var in self.model_vars if 'G_X' in var.name]
        d_Y_vars = [var for var in self.model_vars if 'D_Y' in var.name]
        g_Y_vars = [var for var in self.model_vars if 'G_Y' in var.name]

        self.D_X_trainer = optimizer.minimize(D_X_loss, var_list=d_X_vars)
        self.D_Y_trainer = optimizer.minimize(D_Y_loss, var_list=d_Y_vars)
        self.G_X_trainer = optimizer.minimize(G_loss, var_list=g_X_vars)
        self.G_Y_trainer = optimizer.minimize(F_loss, var_list=g_Y_vars)
        
        for var in self.model_vars:
            print(var.name)
            
        # Summary variables for tensorboard
        self.G_X_loss_summ = tf.summary.scalar("G_loss", G_loss)
        self.G_Y_loss_summ = tf.summary.scalar("F_loss", F_loss)
        self.D_X_loss_summ = tf.summary.scalar("D_X_loss", D_X_loss)
        self.D_Y_loss_summ = tf.summary.scalar("D_Y_loss", D_Y_loss)

        
    def save_images(self, sess, epoch):
        """
        Saves input and output images.
        :param sess: The session.
        :param epoch: Currnt epoch.
        """
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        names = ['inputX_', 'inputY_', 'fakeX_',
                 'fakeY_', 'cycX_', 'cycY_']

        with open(os.path.join(
                self._output_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                inputs = sess.run(self.inputs)
                fake_X_temp, fake_Y_temp, cyc_X_temp, cyc_Y_temp = sess.run([
                    self.fake_images_x,
                    self.fake_images_y,
                    self.cycle_images_x,
                    self.cycle_images_y], 
                    feed_dict={
                    self.input_x: inputs['images_i'],
                    self.input_y: inputs['images_j']
                })

                tensors = [inputs['images_i'], inputs['images_j'],
                           fake_Y_temp, fake_X_temp, cyc_X_temp, cyc_Y_temp]

                for name, tensor in zip(names, tensors):
                    image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                    imsave(os.path.join(self._images_dir, image_name),
                           ((tensor[0] + 1) * 127.5).astype(np.uint8))
                    v_html.write(
                        "<img src=\"" +
                        os.path.join('imgs', image_name) + "\">"
                    )
                v_html.write("<br>")
        
     
    def fake_image_pool(self, num_fakes, fake, fake_pool):
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake
    
    
    def train(self):
        """Training Function."""
        # Load Dataset from the dataset folder
        # Need to be modified
        self.inputs = data_loader.load_data(
            self._dataset_name, self._size_before_crop,
            True, self._do_flipping)

        # Build the network
        self.model()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver()

        max_images = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]

        with tf.Session() as sess:
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step):
                print("In the epoch ", epoch)
                saver.save(sess, os.path.join(self._output_dir, "cyclegan"), global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                if epoch < 100:
                    curr_lr = self._base_lr
                else:
                    curr_lr = self._base_lr - self._base_lr * (epoch - 100) / 100

                self.save_images(sess, epoch)

                for i in range(0, max_images):
                    print("Processing batch {}/{}".format(i, max_images))

                    inputs = sess.run(self.inputs)

                    # Optimizing the G_X network
                    _, fake_Y_temp, summary_str = sess.run(
                        [self.G_X_trainer,
                         self.fake_images_y,
                         self.G_X_loss_summ],
                        feed_dict={
                            self.input_x:
                                inputs['images_i'],
                            self.input_y:
                                inputs['images_j'],
                            self.learning_rate: curr_lr
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_Y_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_Y_temp, self.fake_images_Y)

                    # Optimizing the D_Y network
                    _, summary_str = sess.run(
                        [self.D_Y_trainer, self.D_Y_loss_summ],
                        feed_dict={
                            self.input_x:
                                inputs['images_i'],
                            self.input_y:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_Y: fake_Y_temp1
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    # Optimizing the G_Y network
                    _, fake_X_temp, summary_str = sess.run(
                        [self.G_Y_trainer,
                         self.fake_images_x,
                         self.G_Y_loss_summ],
                        feed_dict={
                            self.input_x:
                                inputs['images_i'],
                            self.input_y:
                                inputs['images_j'],
                            self.learning_rate: curr_lr
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_X_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_X_temp, self.fake_images_X)

                    # Optimizing the D_X network
                    _, summary_str = sess.run(
                        [self.D_X_trainer, self.D_X_loss_summ],
                        feed_dict={
                            self.input_x:
                                inputs['images_i'],
                            self.input_y:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_X: fake_X_temp1
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    writer.flush()
                    self.num_fake_inputs += 1

                sess.run(tf.assign(self.global_step, epoch + 1))

            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)
            
    def test(self):
        """Test Function."""
        print("Testing the results")

        self.inputs = data_loader.load_data(
            self._dataset_name, self._size_before_crop,
            False, self._do_flipping)

        self.model()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            print(self._checkpoint_dir)
            chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
            print(chkpt_fname)
            saver.restore(sess, chkpt_fname)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self._num_imgs_to_save = cyclegan_datasets.DATASET_TO_SIZES[
                self._dataset_name]
            self.save_images(sess, 0)

            coord.request_stop()
            coord.join(threads)


# In[ ]:




