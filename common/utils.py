import numpy as np


def img_to_net_input(img):
    input = np.moveaxis(img, 2, 0)
    input = np.expand_dims(input, axis=0)

    return input