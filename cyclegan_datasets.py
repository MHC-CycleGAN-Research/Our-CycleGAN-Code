"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'fake2real_train': 5965,
    'fake2real_test': 88
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'fake2real_train': '.jpg',
    'fake2real_test': '.jpg',
}

"""The segmentation masks types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_SEGTYPE = {
    'fake2real_train': '.png',
    'fake2real_test': '.png',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'fake2real_train': './input/fake2real/fake2real_train.csv',
    'fake2real_test': './input/fake2real/fake2real_test.csv',
}
