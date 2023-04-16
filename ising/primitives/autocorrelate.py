import jax.numpy as jnp
from jax import Array
from jax.scipy.signal import correlate


def autocorrelate(array: Array, normalise: bool = True) -> Array:
    """
    Autocorrelation function using JAX implementation of SciPy's
    cross-correlation function.
    """
    assert array.ndim == 1

    full_autocorrelation = correlate(array, array, mode="full", method="auto")
    offset = full_autocorrelation.size // 2

    autocorrelation = full_autocorrelation[offset:]

    if normalise:
        autocorrelation = autocorrelation / jnp.max(autocorrelation)

    return autocorrelation
