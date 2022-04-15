from pytest import fixture
from ising.primitives import get_nearest_neighbours

import jax.numpy as jnp


@fixture
def state():
    state = jnp.arange(1, 10)
    state = jnp.reshape(state, (3, 3))

    return state


def test_get_nearest_neighbours_constant(state):
    assert jnp.array_equal(
        get_nearest_neighbours(state, (0, 0), bc_mode="constant", bc_mode_value=0),
        jnp.array([4, 0, 2, 0]),
    )

def test_get_nearest_neighbours_periodic(state):

    assert jnp.array_equal(
        get_nearest_neighbours(state, (0, 0), bc_mode="periodic"),
        jnp.array([4, 7, 2, 3]),
    )

    assert jnp.array_equal(
        get_nearest_neighbours(state, (2, 2), bc_mode="periodic"),
        jnp.array([3, 6, 7, 8]),
    )
