import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf

import cyclegan_datasets
import utils
import random


def _load_samples(csv_name, image_type, is_segmented, seg_type = None):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    if is_segmented is True:
        record_defaults = [tf.constant([], dtype=tf.string),
                           tf.constant([], dtype=tf.string),
                           tf.constant([], dtype=tf.string),
                           tf.constant([], dtype=tf.string)]
        filename_i, filename_j, filename_seg_i, filename_seg_j = tf.decode_csv(
            csv_filename, record_defaults=record_defaults)
        file_segs_i = tf.read_file(filename_seg_i)
        file_segs_j = tf.read_file(filename_seg_j)

    else:
        record_defaults = [tf.constant([], dtype=tf.string),
                           tf.constant([], dtype=tf.string)]
        filename_i, filename_j = tf.decode_csv(
            csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)

    if image_type == '.jpg':
        image_decoded_A = tf.image.decode_jpeg(file_contents_i, channels=utils.IMG_CHANNELS)
        image_decoded_B = tf.image.decode_jpeg(file_contents_j, channels=utils.IMG_CHANNELS)

    elif image_type == '.png':
        image_decoded_A = tf.image.decode_png(file_contents_i, channels=utils.IMG_CHANNELS, dtype=tf.uint8)
        image_decoded_B = tf.image.decode_png(file_contents_j, channels=utils.IMG_CHANNELS, dtype=tf.uint8)

    if is_segmented is True:
        if seg_type == '.jpg':
            seg_decoded_A = tf.image.decode_jpeg(file_segs_i, channels=1)
            seg_decoded_B = tf.image.decode_jpeg(file_segs_j, channels=1)

        elif seg_type == '.png':
            seg_decoded_A = tf.image.decode_png(file_segs_i, channels=1, dtype=tf.uint8)
            seg_decoded_B = tf.image.decode_png(file_segs_j, channels=1, dtype=tf.uint8)

    if is_segmented is True:
        return image_decoded_A, image_decoded_B, seg_decoded_A, seg_decoded_B
    else:
        return image_decoded_A, image_decoded_B


def largest_square_finder(image):
    # find the largest square edge length in a black circle
    h, w = image.shape[-3], image.shape[-2]

    gray = tf.squeeze(tf.image.rgb_to_grayscale(image))
    zeros = tf.zeros_like(gray)
    mask = tf.greater(gray, zeros)
    coordinates_pred = tf.cast(tf.where(mask), tf.float32)

    xy_min = tf.reduce_min(coordinates_pred, axis=0)
    xy_max = tf.reduce_max(coordinates_pred, axis=0)
    diameter = tf.reduce_min(tf.subtract(xy_max,xy_min), axis=0)
    square_edge =  tf.cast(tf.floordiv(diameter,tf.math.sqrt(2.0)),tf.int64)

    return square_edge

def crop_center(image, square_edge):
    # crop the center part of the image defined by square edge
    h, w = image.shape[-3], image.shape[-2]

    cropped_image = tf.image.crop_to_bounding_box(image, 
        (h-square_edge) // 2, (w-square_edge) // 2, square_edge, square_edge)
    image = tf.image.resize_images(cropped_image, [h, w])

    return image

def random_crop_image_and_seg(image, seg):
    # randomly crop the the images and corresponding segmentation masks
    h, w = image.shape[-3], image.shape[-2]

    crop_h = random.randint(0,h-utils.IMG_HEIGHT-1)
    crop_w = random.randint(0, w-utils.IMG_WIDTH-1)
    image = tf.image.crop_to_bounding_box(image, crop_h, crop_w, utils.IMG_HEIGHT, utils.IMG_WIDTH)
    seg = tf.image.crop_to_bounding_box(seg, crop_h, crop_w, utils.IMG_HEIGHT, utils.IMG_WIDTH)

    return image, seg

 
def random_flip_left_right_image_and_seg(image, seg):
    # randomly crop the the image and the corresponding segmentation mask
    threshold = random.uniform(0.0,1.0)
    if threshold > 0.5:
        image = tf.image.flip_left_right(image)
        seg = tf.image.flip_left_right(seg)

    return image, seg

def load_data(dataset_name, image_size_before_crop, do_shuffle=True, 
    do_flipping=False, do_ccropping=False, do_rcropping=False, is_segmented=False):
    """

    :param dataset_name: The name of the dataset.
    :param image_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:
    """
    if dataset_name not in cyclegan_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)

    csv_name = cyclegan_datasets.PATH_TO_CSV[dataset_name]

    if is_segmented is True:
        image_i, image_j, seg_i, seg_j = _load_samples(
            csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name], 
            is_segmented, cyclegan_datasets.DATASET_TO_SEGTYPE[dataset_name])
        seg_i = tf.image.resize_images(seg_i, [image_size_before_crop, image_size_before_crop])
        seg_j = tf.image.resize_images(seg_j, [image_size_before_crop, image_size_before_crop])
    else:
        image_i, image_j = _load_samples(
            csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name], is_segmented)

    # Preprocessing:
    image_i = tf.image.resize_images(image_i, [image_size_before_crop, image_size_before_crop])
    image_j = tf.image.resize_images(image_j, [image_size_before_crop, image_size_before_crop])

    if do_ccropping is True:
        edge_i = largest_square_finder(image_i)
        edge_j = largest_square_finder(image_j)
        image_i = crop_center(image_i, edge_i)
        image_j = crop_center(image_j, edge_j)
        if is_segmented is True:
            seg_i = crop_center(seg_i, edge_i)
            seg_j = crop_center(seg_j, edge_j)

    if do_flipping is True:
        if is_segmented is True:
            image_i, seg_i = random_flip_left_right_image_and_seg(image_i, seg_i)
            image_j, seg_j = random_flip_left_right_image_and_seg(image_j, seg_j)
        else:
            image_i = tf.image.random_flip_left_right(image_i)
            image_j = tf.image.random_flip_left_right(image_j)

    if do_rcropping is True:
        if is_segmented is True:
            image_i, seg_i = random_crop_image_and_seg(image_i, seg_i)
            image_j, seg_j = random_crop_image_and_seg(image_j, seg_j)    
        else:
            image_i = tf.random_crop(image_i, [utils.IMG_HEIGHT, utils.IMG_WIDTH, 3])
            image_j = tf.random_crop(image_j, [utils.IMG_HEIGHT, utils.IMG_WIDTH, 3])

    image_i = tf.subtract(tf.div(image_i, 127.5), 1)
    image_j = tf.subtract(tf.div(image_j, 127.5), 1)

    # Batch
    if is_segmented is True:
        if do_shuffle is True:
            images_i, images_j, segs_i, segs_j = tf.train.shuffle_batch([image_i, image_j, seg_i, seg_j], 1, 5000, 100)
        else:
            images_i, images_j, segs_i, segs_j = tf.train.batch([image_i, image_j, seg_i, seg_j], 1)
        inputs = { 'images_i': images_i, 'images_j': images_j, 'segs_i': segs_i, 'segs_j': segs_j }

    else:
        if do_shuffle is True:
            images_i, images_j = tf.train.shuffle_batch([image_i, image_j], 1, 5000, 100)
        else:
            images_i, images_j = tf.train.batch([image_i, image_j], 1)
        inputs = { 'images_i': images_i, 'images_j': images_j }
        
    return inputs
