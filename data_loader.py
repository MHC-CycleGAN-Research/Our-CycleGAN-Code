import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf

import cyclegan_datasets
import utils


def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)

    if image_type == '.jpg':
        image_decoded_A = tf.image.decode_jpeg(
            file_contents_i, channels=utils.IMG_CHANNELS)
        image_decoded_B = tf.image.decode_jpeg(
            file_contents_j, channels=utils.IMG_CHANNELS)
    elif image_type == '.png':
        image_decoded_A = tf.image.decode_png(
            file_contents_i, channels=utils.IMG_CHANNELS, dtype=tf.uint8)
        image_decoded_B = tf.image.decode_png(
            file_contents_j, channels=utils.IMG_CHANNELS, dtype=tf.uint8)

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


def load_data(dataset_name, image_size_before_crop,
              do_shuffle=True, do_flipping=False, do_ccropping=False, do_rcropping=False):
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

    image_i, image_j = _load_samples(
        csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])

    # Preprocessing:
    image_i = tf.image.resize_images(image_i, [image_size_before_crop, image_size_before_crop])
    image_j = tf.image.resize_images(image_j, [image_size_before_crop, image_size_before_crop])

    if do_ccropping is True:
        image_i = crop_center(image_i, largest_square_finder(image_i))
        image_j = crop_center(image_j, largest_square_finder(image_j))

    if do_flipping is True:
        image_i = tf.image.random_flip_left_right(image_i)
        image_j = tf.image.random_flip_left_right(image_j)

    if do_rcropping is True:
        image_i = tf.random_crop(
            image_i, [utils.IMG_HEIGHT, utils.IMG_WIDTH, 3])
        image_j = tf.random_crop(
            image_j, [utils.IMG_HEIGHT, utils.IMG_WIDTH, 3])

    image_i = tf.subtract(tf.div(image_i, 127.5), 1)
    image_j = tf.subtract(tf.div(image_j, 127.5), 1)

    # Batch
    if do_shuffle is True:
        images_i, images_j = tf.train.shuffle_batch(
           [image_i, image_j], 1, 5000, 100)
    else:
        images_i, images_j = tf.train.batch(
           [image_i, image_j], 1)
    inputs = {
        'images_i': images_i,
        'images_j': images_j
    }
        
    return inputs
