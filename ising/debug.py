from jax.debug import print as jax_print


def dbg(*args, **kwargs):
    return jax_print(*args, **kwargs)
