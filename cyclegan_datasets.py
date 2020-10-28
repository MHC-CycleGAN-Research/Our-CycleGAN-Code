"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'day2night_train': 50,
    'day2night_test': 5
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'day2night_train': '.jpg',
    'day2night_test': '.jpg',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'day2night_train': './input/day2night/day2night_train.csv',
    'day2night_test': './input/day2night/day2night_test.csv',
}
