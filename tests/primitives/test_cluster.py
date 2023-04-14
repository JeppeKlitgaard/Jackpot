import numpy as np
from numpy.testing import assert_array_equal

from ising.primitives.convolve import convolve_with_wrapping


def test_convolve_with_wrapping():
    array = np.zeros((5, 5), dtype=int)
    array[1, 0] = 1
    kernel = np.asarray(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    convolved = convolve_with_wrapping(array, kernel)

    should_be = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    assert_array_equal(convolved, should_be)
