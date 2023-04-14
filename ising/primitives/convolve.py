import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.signal import convolve


def convolve_with_wrapping(array: Array, kernel: Array) -> Array:
    assert array.ndim == kernel.ndim
    pad_sizes = np.asarray(kernel.shape) // 2

    padded_shape = tuple(
        ax + pad_size * 2 for ax, pad_size in zip(array.shape, pad_sizes)
    )
    padded_array = jnp.zeros(padded_shape, dtype=array.dtype)
    # Add array to padded array
    middle_slice = tuple(
        slice(pad_size, dim + pad_size) for pad_size, dim in zip(pad_sizes, array.shape)
    )
    padded_array = padded_array.at[middle_slice].set(array)

    # Add 'sides'
    for i in range(array.ndim):
        slicer1_from = tuple(
            slice(dim + pad_size - pad_size, dim + 2 * pad_size - pad_size)
            if i == j
            else slice(None)
            for j, (pad_size, dim) in enumerate(zip(pad_sizes, array.shape))
        )
        slicer1_to = tuple(
            slice(0, pad_size) if i == j else slice(None)
            for j, pad_size in enumerate(pad_sizes)
        )

        slicer2_from = tuple(
            slice(pad_size, 2 * pad_size) if i == j else slice(None)
            for j, pad_size in enumerate(pad_sizes)
        )
        slicer2_to = tuple(
            slice(dim + pad_size, dim + 2 * pad_size) if i == j else slice(None)
            for j, (pad_size, dim) in enumerate(zip(pad_sizes, array.shape))
        )
        padded_array.at[slicer1_to].set(padded_array[slicer1_from])
        padded_array.at[slicer2_to].set(padded_array[slicer2_from])

    convolved_padded = convolve(padded_array, kernel, mode="same")
    convolved = convolved_padded[middle_slice].astype(array.dtype)

    return convolved
