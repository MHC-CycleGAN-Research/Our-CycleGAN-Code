#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import json
import model

import click

@click.command()
@click.option('--to_train',
              type=click.INT,
              default=0,
              help='1=training; 2=resuming from latest checkpoint; 0=testing.')
@click.option('--log_dir',
              type=click.STRING,
              default='./output/cyclegan/exp_01',
              help='The path to save the training log.')
@click.option('--config_filename',
              type=click.STRING,
              default='./configs/exp_01.json',
              help='The path to the dataset configuration file.')
@click.option('--checkpoint_dir',
              type=click.STRING,
              default='./output/cyclegan/exp_01/20201012-175320',
              help='The path to save model checkpoints.')
@click.option('--skip',
              type=click.BOOL,
              default=False,
              help='Whether to skip a few nodes during training.')

def main(to_train, log_dir, config_filename, checkpoint_dir, skip):
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

