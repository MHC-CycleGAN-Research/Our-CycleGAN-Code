"""Create datasets for training and testing."""
import csv
import os
import random

import click

import cyclegan_datasets


def create_list(foldername, fulldir=True, suffix=".jpg"):
    """

    :param foldername: The full path of the folder.
    :param fulldir: Whether to return the full path or not.
    :param suffix: Filter by suffix.

    :return: The list of filenames in the folder with given suffix.

    """
    file_list_tmp = os.listdir(foldername)
    file_list = []
    if fulldir is True :
        for item in file_list_tmp:
            if item.endswith(suffix):                                                              
                file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    file_list.sort()
    return file_list


@click.command()
@click.option('--image_path_a',
              type=click.STRING,
              default='./input/fake2real/trainA',
              help='The path to the images from domain_a.')
@click.option('--image_path_b',
              type=click.STRING,
              default='./input/fake2real/trainB',
              help='The path to the images from domain_b.')
@click.option('--image_seg_a',
              type=click.STRING,
              default='./input/fake2real/trainSegA',
              help='The path to the segmentation masks from domain_a.')
@click.option('--image_seg_b',
              type=click.STRING,
              default='./input/fake2real/trainSegB',
              help='The path to the segmentation masks from domain_b.')
@click.option('--dataset_name',
              type=click.STRING,
              default='fake2real_train',
              help='The name of the dataset in cyclegan_dataset.')
@click.option('--do_shuffle',
              type=click.BOOL,
              default=True,
              help='Whether to shuffle images when creating the dataset.')
@click.option('--is_segmented',
              type=click.BOOL,
              default=True,
              help='Whether segmentation masks are provided.')

def create_dataset(image_path_a, image_path_b, image_seg_a, image_seg_b,
                   dataset_name, do_shuffle, is_segmented):
    list_a = create_list(image_path_a, True,
                         cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    list_b = create_list(image_path_b, True,
                         cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    if is_segmented is True:
      list_seg_a = create_list(image_seg_a, True,
                           cyclegan_datasets.DATASET_TO_SEGTYPE[dataset_name])
      list_seg_b = create_list(image_seg_b, True,
                           cyclegan_datasets.DATASET_TO_SEGTYPE[dataset_name])

    output_path = cyclegan_datasets.PATH_TO_CSV[dataset_name]

    num_rows = cyclegan_datasets.DATASET_TO_SIZES[dataset_name]

    sequence_a = [item for item in range(num_rows)]
    sequence_b = [item for item in range(num_rows)]
    if do_shuffle is True:
        random.shuffle(sequence_a)
        random.shuffle(sequence_b)      

    all_data_tuples = []

    if is_segmented is True:
      for i in range(num_rows):
          all_data_tuples.append((
              list_a[sequence_a[i] % len(list_a)],
              list_b[sequence_b[i] % len(list_b)],
              list_seg_a[sequence_a[i] % len(list_seg_a)],
              list_seg_b[sequence_b[i] % len(list_seg_b)]
          ))
    else:
      for i in range(num_rows):
          all_data_tuples.append((
              list_a[i % len(list_a)],
              list_b[i % len(list_b)]
          ))

    with open(output_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for data_tuple in enumerate(all_data_tuples):
            csv_writer.writerow(list(data_tuple[1]))


if __name__ == '__main__':
    create_dataset()
