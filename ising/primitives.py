"""
This file contains 'primitives' for the `ising` package.

Primitives are jittable pure functions.
"""
from jax import lax
from jax import random
from jax import jit, pmap, soft_pmap, config, vmap
from jax.experimental.maps import xmap
import jax.numpy as jnp
import numpy as np
from ising.typing import TSpins, TSpin

config.enable_omnistaging()


def get_random_point_idx(
    rng_key: int, dimensionality: int, size: int
) -> tuple[int, ...]:
    return tuple(random.randint(rng_key, (dimensionality,), minval=0, maxval=size))


get_random_point_idx = jit(
    get_random_point_idx, static_argnames=("dimensionality", "size")
)


@jit
def get_nearest_neighbours(state: TSpins, idx: tuple[int]) -> TSpins:
    """
    NOTE: Uses periodic boundary conditions.
    """
    nearest_neighbours = []
    for n in range(state.ndim):
        for delta in [1, -1]:
            selector = jnp.array(idx)
            selector = selector.at[n].add(delta)
            slice_ = tuple(selector)

            neighbour = state[slice_]

            nearest_neighbours.append(neighbour)

    nn: TSpins = jnp.array(nearest_neighbours)

    return nn


def get_hamiltonian_delta(
    state: TSpins,
    idx: tuple[int, ...],
    trial_spin: TSpin,
    interaction_bilinear: float,
    interaction_biquadratic: float,
    interaction_anisotropy: float,
    interaction_bicubic: float,
    interaction_external_field: float,
) -> float:
    """
    Calculates the Hamiltonian Delta by only considering nearest neighbours.
    This is much more efficient than calculating the Hamiltonian for each
    Metropolis step.

    Not that interaction coefficients are JIT compile-static and thus any
    Hamiltonian contributions with interaction coefficients that are zero
    are automatically discarded during JIT tree-shaking.

    Introduction of if statements to do manual tree-shaking would at best
    slow down tracing process and at worst lead to slower runtimes.
    """
    H = 0.0

    current_spin = state[idx]
    delta_spin = trial_spin - current_spin

    neighbours = get_nearest_neighbours(state, idx)

    neighbours_sq = jnp.square(neighbours)
    delta_spin_sq = jnp.square(delta_spin)

    # J - Calculate bilinear exchange energy (nearest neighbour)
    H -= interaction_bilinear * (delta_spin * neighbours).sum()

    # K - Calculate biquadratic exchange energy (nearest neighbour)
    H -= interaction_biquadratic * (delta_spin_sq * neighbours_sq).sum()

    # D - Calculate anisotropy energy
    H -= interaction_anisotropy * delta_spin_sq

    # L - Calculate bicubic exchange energy (nearest neighbour)
    H -= (
        interaction_bicubic
        * (delta_spin_sq * neighbours + delta_spin * neighbours_sq).sum()
    )

    # H - Calculate external field energy
    H -= interaction_external_field * delta_spin

    return H


get_hamiltonian_delta = jit(
    get_hamiltonian_delta,
    static_argnames=(
        "interaction_bilinear",
        "interaction_biquadratic",
        "interaction_anisotropy",
        "interaction_bicubic",
        "interaction_external_field",
    ),
)


def _run_mcmc_step(
    rng_key: int,
    state: TSpins,
    possible_states: TSpins,
    beta: float,
    interaction_bilinear: float,
    interaction_biquadratic: float,
    interaction_anisotropy: float,
    interaction_bicubic: float,
    interaction_external_field: float,
) -> TSpins:
    point_key, spin_key, boltzmann_key = random.split(rng_key, 3)

    idx = get_random_point_idx(point_key, state.ndim, state.shape[0])

    trial_spin = random.choice(spin_key, possible_states)
    H_delta = get_hamiltonian_delta(
        state,
        idx,
        trial_spin,
        interaction_bilinear,
        interaction_biquadratic,
        interaction_anisotropy,
        interaction_bicubic,
        interaction_external_field,
    )

    # Change if new energy is lower
    # Or if higher but boltzmann says we should
    # Else return old state
    def if_energy_lower(
        state: TSpins, idx: tuple[int, ...], trial_spin: TSpin
    ) -> TSpins:
        return state.at[idx].set(trial_spin)

    def if_energy_higher(
        state: TSpins, idx: tuple[int, ...], trial_spin: TSpin
    ) -> TSpins:
        return lax.cond(
            jnp.exp(-beta * H_delta) > random.uniform(boltzmann_key),
            lambda s, i, t: s.at[i].set(t),
            lambda s, i, t: s,
            state,
            idx,
            trial_spin,
        )

    return lax.cond(
        H_delta < 0, if_energy_lower, if_energy_higher, state, idx, trial_spin
    )


run_mcmc_step = jit(
    _run_mcmc_step,
    static_argnames=(
        "beta",
        "interaction_bilinear",
        "interaction_biquadratic",
        "interaction_anisotropy",
        "interaction_bicubic",
        "interaction_external_field",
    ),
)


def _run_mcmc_steps(
    steps: int,
    rng_key: int,
    state: TSpins,
    possible_states: TSpins,
    beta: float,
    interaction_bilinear: float,
    interaction_biquadratic: float,
    interaction_anisotropy: float,
    interaction_bicubic: float,
    interaction_external_field: float,
) -> TSpins:

    keys = random.split(rng_key, steps)

    def body_fun(i: int, state: TSpins) -> TSpins:
        return _run_mcmc_step(
            keys[i],
            state,
            possible_states,
            beta,
            interaction_bilinear,
            interaction_biquadratic,
            interaction_anisotropy,
            interaction_bicubic,
            interaction_external_field,
        )

    return lax.fori_loop(0, steps, body_fun, state)


run_mcmc_steps = jit(
    _run_mcmc_steps,
    static_argnames=(
        "steps",
        # "beta",
        "interaction_bilinear",
        "interaction_biquadratic",
        "interaction_anisotropy",
        "interaction_bicubic",
        "interaction_external_field",
    ),
)

multi_run_mcmc_steps = xmap(
    run_mcmc_steps,
    in_axes=(None, None, 0, None, 0, None, None, None, None, None),
    out_axes=(0)
    # static_broadcasted_argnums=(0, 5, 6, 7, 8, 9),
)
# def multistate_run_mcmc_steps(
#     steps_tuple: tuple[int, ...],
#     rng_key_tuple: [int,
#     state: TSpins,
#     possible_states: TSpins,
#     beta: float,
#     interaction_bilinear: float,
#     interaction_biquadratic: float,
#     interaction_anisotropy: float,
#     interaction_bicubic: float,
#     interaction_external_field: float,
# ) -> TSpins:
