#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import json
import model

def main(to_train=1, log_dir="./output/cyclegan/exp_01", config_filename="./configs/exp_01.json", checkpoint_dir="./output/cyclegan/checkpoint_01", skip=False):
    """
    :param to_train: Specify whether it is training or testing. 1: training; 2:
     resuming from latest checkpoint; 0: testing.
    :param log_dir: The root dir to save checkpoints and imgs. The actual dir
    is the root dir appended by the folder with the name timestamp.
    :param config_filename: The configuration file.
    :param checkpoint_dir: The directory that saves the latest checkpoint. It
    only takes effect when to_train == 2.
    :param skip: A boolean indicating whether to add skip connection between
    input and output.
    """
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    lambda_a = float(config['_LAMBDA_A']) if '_LAMBDA_A' in config else 10.0
    lambda_b = float(config['_LAMBDA_B']) if '_LAMBDA_B' in config else 10.0
    pool_size = int(config['pool_size']) if 'pool_size' in config else 50

    to_restore = (to_train == 2)
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 200
    network_version = str(config['network_version'])
    dataset_name = str(config['dataset_name'])
    do_flipping = bool(config['do_flipping'])

    cyclegan_model = model.CycleGAN(pool_size, lambda_a, lambda_b, log_dir,
                              to_restore, base_lr, max_step,
                              dataset_name, checkpoint_dir, do_flipping, skip)

    if to_train > 0:
        cyclegan_model.train()
    else:
        cyclegan_model.test()


if __name__ == '__main__':
    main()

