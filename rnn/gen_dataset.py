import tf.keras
import tf.layers
import tf.estimator
import tf.data
import numpy as np


test_lens = [2, 2, 3, 1, 1]
test_offrs = (
    [
        [
            [1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]
        ],
        [
            [19, 11, 45, 76], [120, 42, 19, 19], [0, 0, 0, 0]
        ],
        [
            [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
        ],
        [
            [20, 22, 21, 24], [0, 0, 0, 0], [0, 0, 0, 0]
        ],
        [
            [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]
        ]
    ])
# transform so batch is the first index


def input_fun(features, )
