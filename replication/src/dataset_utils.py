import numpy as np

class DatasetUtils:

    def __init__(self):
        pass

    @staticmethod
    def apply_padding(image, desired_height, desired_width):
        """
        Applies padding to the 3d image, only in the height and width directions.
        The number of stacks stays the same.
        """
        height_padding = (desired_height - image.shape[1]) // 2
        width_padding = (desired_width - image.shape[2]) // 2

        height_padding_extra = (desired_height - image.shape[1]) % 2
        width_padding_extra = (desired_width - image.shape[2]) % 2

        padding_sizes = [(0, 0),
                         (height_padding, height_padding + height_padding_extra),
                         (width_padding, width_padding + width_padding_extra)]

        padded_image = np.pad(image, padding_sizes, mode='reflect')

        return padded_image
