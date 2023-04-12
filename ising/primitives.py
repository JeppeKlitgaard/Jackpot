"""
This file contains 'primitives' for the `ising` package.

Primitives are jittable pure functions.
"""
from jax import Array
from jax import lax
from jax import random
from jax import jit, pmap, vmap
from jax.random import KeyArray
import jax.numpy as jnp
from ising.typing import TSpins, TSpin, ScalarFloat, RNGKey
from typing import Any, Literal
from functools import partial
from jax.scipy.signal import convolve
from scipy import constants

from ising.typing import RNGKey, ScalarFloat, TIndex, TShape, TSpin, TSpins

TBCModes = Literal["constant", "periodic"]


def get_random_point_idx2(rng_key: RNGKey, shape: TShape) -> TIndex:
    dim = len(shape)

    minvals = np.zeros(dim)
    maxvals = np.asarray(shape)

    idx = random.randint(key=rng_key, shape=(dim,), minval=minvals, maxval=maxvals)

    return idx

get_random_point_idx2 = jit(
    get_random_point_idx2, static_argnames=("shape")
)



def temperature_to_beta(temperature_or_temperatures: ScalarFloat) -> ScalarFloat:
    reciprocal: ScalarFloat = constants.Boltzmann * temperature_or_temperatures
    return 1.0 / reciprocal


def get_random_point_idx(
    rng_key: KeyArray, dimensionality: int, size: int
) -> TIndex:
    return tuple(random.randint(rng_key, (dimensionality,), minval=0, maxval=size))


get_random_point_idx = jit(
    get_random_point_idx, static_argnames=("dimensionality", "size")
)


@partial(jit, static_argnames=("bc_mode", "bc_mode_value"))
def get_nearest_neighbours(
    state: TSpins,
    idx: tuple[int],
    bc_mode: TBCModes,
    bc_mode_value: float | None = None,
) -> TSpins:
    """
    Boundary condition: OOB are set to 0.
    """
    nearest_neighbours = []
    for n in range(state.ndim):
        for delta in [1, -1]:
            selector = jnp.array(idx)
            selector = selector.at[n].add(delta)

            # Padding with zeros
            if bc_mode == "constant":
                assert bc_mode_value is not None

                selector = jnp.where(selector == -1, selector.size + 1, selector)
                neighbour = state.at[tuple(selector)].get(
                    mode="fill", fill_value=bc_mode_value
                )

            elif bc_mode == "periodic":
                selector = jnp.where(selector == selector.size + 1, 0, selector)
                neighbour = state.at[tuple(selector)].get()

            nearest_neighbours.append(neighbour)

    nn: TSpins = jnp.array(nearest_neighbours)

    return nn


@partial(
    jit,
    static_argnames=(
        "interaction_bilinear",
        "interaction_biquadratic",
        "interaction_anisotropy",
        "interaction_bicubic",
        "interaction_external_field",
        "nuclear_magnetic_moment",
        "bc_mode",
        "bc_mode_value",
    ),
)
def get_hamiltonian_delta(
    state: TSpins,
    idx: TIndex,
    trial_spin: TSpin,
    interaction_bilinear: float,
    interaction_biquadratic: float,
    interaction_anisotropy: float,
    interaction_bicubic: float,
    interaction_external_field: float,
    nuclear_magnetic_moment: float,
    bc_mode: TBCModes,
    bc_mode_value: float | None = None,
) -> ScalarFloat:
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
    H: ScalarFloat = 0.0

    current_spin = state[idx]
    delta_spin = trial_spin - current_spin

    neighbours = get_nearest_neighbours(
        state, idx, bc_mode=bc_mode, bc_mode_value=bc_mode_value
    )

    neighbours_sq = jnp.square(neighbours)
    delta_spin_sq = jnp.square(delta_spin)

    # J - Calculate bilinear exchange energy (nearest neighbour)
    H -= 2 * interaction_bilinear * (delta_spin * neighbours).sum()

    # K - Calculate biquadratic exchange energy (nearest neighbour)
    H -= 2 * interaction_biquadratic * (delta_spin_sq * neighbours_sq).sum()

    # D - Calculate anisotropy energy
    H -= interaction_anisotropy * delta_spin_sq

    # L - Calculate bicubic exchange energy (nearest neighbour)
    H -= (
        interaction_bicubic
        * (delta_spin_sq * neighbours + delta_spin * neighbours_sq).sum()
    )

    # H - Calculate external field energy
    H -= nuclear_magnetic_moment * interaction_external_field * delta_spin

    return H


@partial(
    jit,
    static_argnames=(
        "nn_kernel",
        "interaction_bilinear",
        "interaction_biquadratic",
        "interaction_anisotropy",
        "interaction_bicubic",
        "interaction_external_field",
        "nuclear_magnetic_moment",
    ),
)
def get_hamiltonian(
    state: TSpins,
    nn_kernel: TSpins,
    interaction_bilinear: float,
    interaction_biquadratic: float,
    interaction_anisotropy: float,
    interaction_bicubic: float,
    interaction_external_field: float,
    nuclear_magnetic_moment: float,
) -> ScalarFloat:
    kernel = jnp.asarray(nn_kernel)

    H: ScalarFloat = jnp.asarray(0.0)
    state_sq = jnp.square(state)

    # J - Calculate bilinear exchange energy (nearest neighbour)
    H -= interaction_bilinear * (state * convolve(state, kernel, mode="same")).sum()

    # K - Calculate biquadratic exchange energy (nearest neighbour)
    H -= (
        interaction_biquadratic
        * (state_sq * convolve(state_sq, kernel, mode="same")).sum()
    )

    # D - Calculate anisotropy energy
    H -= interaction_anisotropy * state_sq.sum()

    # L - Calculate bicubic exchange energy (nearest neighbour)
    H -= (
        interaction_bicubic
        * 2
        * (state * convolve(state_sq, kernel, mode="same")).sum()
    )

    # H - Calculate external field energy
    H -= nuclear_magnetic_moment * interaction_external_field * state.sum()

    return H


@partial(jit, static_argnames=("nuclear_magnetic_moment",))
def get_magnetisation_density(state: TSpins, nuclear_magnetic_moment: float) -> Array:
    return nuclear_magnetic_moment * jnp.sum(state) / jnp.size(state)


@partial(
    jit,
    static_argnames=(
        "interaction_bilinear",
        "interaction_biquadratic",
        "interaction_anisotropy",
        "interaction_bicubic",
        "interaction_external_field",
        "nuclear_magnetic_moment",
        "bc_mode",
        "bc_mode_value",
    ),
)
def run_mcmc_step(
    rng_key: KeyArray,
    state: TSpins,
    possible_states: TSpins,
    beta: float,
    interaction_bilinear: float,
    interaction_biquadratic: float,
    interaction_anisotropy: float,
    interaction_bicubic: float,
    interaction_external_field: float,
    nuclear_magnetic_moment: float,
    bc_mode: TBCModes,
    bc_mode_value: float | None = None,
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
        nuclear_magnetic_moment,
        bc_mode=bc_mode,
        bc_mode_value=bc_mode_value,
    )

    # Change if new energy is lower
    # Or if higher but boltzmann says we should
    # Else return old state
    def if_energy_lower(
        state: TSpins, idx: TIndex, trial_spin: TSpin
    ) -> TSpins:
        return state.at[idx].set(trial_spin)

    def if_energy_higher(
        state: TSpins, idx: TIndex, trial_spin: TSpin
    ) -> TSpins:
        out: Array = lax.cond(
            jnp.exp(-beta * H_delta) > random.uniform(boltzmann_key),
            lambda s, i, t: s.at[i].set(t),
            lambda s, i, t: s,
            state,
            idx,
            trial_spin,
        )
        return out

    out: Array = lax.cond(
        H_delta < 0, if_energy_lower, if_energy_higher, state, idx, trial_spin
    )
    return out


@partial(
    jit,
    static_argnames=(
        "steps",
        "interaction_bilinear",
        "interaction_biquadratic",
        "interaction_anisotropy",
        "interaction_bicubic",
        "interaction_external_field",
        "nuclear_magnetic_moment",
        "bc_mode",
        "bc_mode_value",
    ),
)
def run_mcmc_steps(
    steps: int,
    rng_key: KeyArray,
    state: TSpins,
    possible_states: TSpins,
    beta: float,
    interaction_bilinear: float,
    interaction_biquadratic: float,
    interaction_anisotropy: float,
    interaction_bicubic: float,
    interaction_external_field: float,
    nuclear_magnetic_moment: float,
    bc_mode: TBCModes,
    bc_mode_value: float | None = None,
) -> TSpins:
    keys = random.split(rng_key, steps)

    def body_fun(i: int, state: TSpins) -> TSpins:
        out: Array = run_mcmc_step(
            keys[i],
            state,
            possible_states,
            beta,
            interaction_bilinear,
            interaction_biquadratic,
            interaction_anisotropy,
            interaction_bicubic,
            interaction_external_field,
            nuclear_magnetic_moment,
            bc_mode=bc_mode,
            bc_mode_value=bc_mode_value,
        )
        return out

    out: Array = lax.fori_loop(0, steps, body_fun, state)
    return out


vrun_mcmc_steps = jit(
    vmap(
        run_mcmc_steps,
        in_axes=(None, 0, 0, None, 0, None, None, None, None, None, None, None, None),
    ),
    static_argnames=(
        "steps",
        "interaction_bilinear",
        "interaction_biquadratic",
        "interaction_anisotropy",
        "interaction_bicubic",
        "interaction_external_field",
        "nuclear_magnetic_moment",
        "bc_mode",
        "bc_mode_value",
    ),
)

pvrun_mcmc_steps = pmap(
    vrun_mcmc_steps,
    in_axes=(None, 0, 0, None, 0, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(0, 5, 6, 7, 8, 9, 10, 11, 12),
)


@partial(
    jit,
    static_argnames=(
        "nn_kernel",
        "interaction_bilinear",
        "interaction_biquadratic",
        "interaction_anisotropy",
        "interaction_bicubic",
        "interaction_external_field",
        "nuclear_magnetic_moment",
    ),
)
def get_energy_and_magnetisation(
    state: TSpins,
    nn_kernel: tuple[Any, ...],
    interaction_bilinear: float,
    interaction_biquadratic: float,
    interaction_anisotropy: float,
    interaction_bicubic: float,
    interaction_external_field: float,
    nuclear_magnetic_moment: float,
) -> tuple[float, float]:
    energy = get_hamiltonian(
        state,
        nn_kernel,
        interaction_bilinear,
        interaction_biquadratic,
        interaction_anisotropy,
        interaction_bicubic,
        interaction_external_field,
        nuclear_magnetic_moment,
    )

    magnetisation_density = get_magnetisation_density(state, nuclear_magnetic_moment)

    return energy, magnetisation_density


def get_equilibrium_energy_and_magnetisation(
    state: TSpins,
    rng_key: KeyArray,
    possible_states: TSpins,
    beta: float,
    nn_kernel: TSpins,
    interaction_bilinear: float,
    interaction_biquadratic: float,
    interaction_anisotropy: float,
    interaction_bicubic: float,
    interaction_external_field: float,
    nuclear_magnetic_moment: float,
    bc_mode: TBCModes,
    bc_mode_value: float | None = None,
):
    new_state = run_mcmc_step(
        rng_key,
        state,
        possible_states,
        beta,
        interaction_bilinear,
        interaction_biquadratic,
        interaction_anisotropy,
        interaction_bicubic,
        interaction_external_field,
        nuclear_magnetic_moment,
        bc_mode=bc_mode,
        bc_mode_value=bc_mode_value,
    )

    return get_energy_and_magnetisation(
        new_state,
        nn_kernel,
        interaction_bilinear,
        interaction_biquadratic,
        interaction_anisotropy,
        interaction_bicubic,
        interaction_external_field,
        nuclear_magnetic_moment,
    )


vget_equilibrium_energy_and_magnetisation = vmap(
    vmap(
        get_equilibrium_energy_and_magnetisation,
        in_axes=(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ),
    in_axes=(
        0,
        0,
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
)

pvget_equilibrium_energy_and_magnetisation = pmap(
    vget_equilibrium_energy_and_magnetisation,
    in_axes=(
        0,
        0,
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    static_broadcasted_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12),
)
