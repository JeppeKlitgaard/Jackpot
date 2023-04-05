from __future__ import annotations

from typing import Sequence, cast
import numpy as np
import jax.numpy as jnp
from jax import random
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from scipy.ndimage import generate_binary_structure
from scipy import constants
from ising.typing import TSpins, TSpin, ScalarInt
from ising.primitives import (
    TBCModes,
    get_hamiltonian,
    get_magnetisation_density,
    temperature_to_beta,
    pvrun_mcmc_steps,
    pvget_equilibrium_energy_and_magnetisation,
)

from ising.utils.types import lists_to_tuples


class IsingStateND:
    """
    Represents a state of the ND Ising Model.

    This class offers a high-level interface to the state, but when doing
    high-performance numerical operations the underlying JAX array
    should be used.

    Attributes:
        - arr: Underlying JAX Array
    """

    def __init__(self, arr: TSpins, spin_states: tuple[float, ...]):
        self.arr = arr

        self.spin_states = spin_states

    @property
    def dimensionality(self) -> int:
        return self.arr.ndim

    def copy(self) -> IsingStateND:
        """
        Deepcopy does not work on Jax DeviceArrays currently.

        See: https://github.com/google/jax/issues/2632
        """
        return IsingStateND(self.arr, self.spin_states)

    def plot(self) -> Figure:
        norm = mpl.colors.Normalize(
            vmin=min(self.spin_states), vmax=max(self.spin_states)
        )

        fig = plt.figure()
        fig.set_size_inches(12, 12)

        match self.dimensionality:
            case 2:
                ax = fig.add_subplot()
                im = ax.imshow(self.arr, norm=norm)

            case 3:
                ax = fig.add_subplot(111, projection="3d")

                x, y, z = np.indices(self.arr.shape)
                im = ax.scatter(
                    x.ravel(),
                    y.ravel(),
                    z.ravel(),
                    c=self.arr.ravel(),
                    norm=norm,
                    alpha=0.2,
                )

                ax.axis("off")

            case _:
                raise ValueError(
                    f"State plotting not supported for {self.dimensionality}D states"
                )

        colors = [im.cmap(norm(spin_state)) for spin_state in self.spin_states]
        patches = [
            mpatches.Patch(color=colors[i], label=str(spin_state))
            for i, spin_state in enumerate(self.spin_states)
        ]

        plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2)

        info_line = f"$S = {self.arr.sum():.1f}$"
        plt.title(f"Spin states\n{info_line}")

        return fig


class IsingModelND:
    def __init__(
        self,
        dimensionality: int,
        *,
        size: int = 1024,
        spin: float = 0.5,
        spin_states: TSpins | None = None,
        spin_distribution: TSpins | tuple[float, ...] | None = None,
        initial_state: IsingStateND | None = None,
        rng_seed: int = 0,
        interaction_bilinear: float = 1.0,
        interaction_biquadratic: float = 0.0,
        interaction_anisotropy: float = 0.0,
        interaction_bicubic: float = 0.0,
        interaction_external_field: float = 0.0,
        nuclear_magnetic_moment: float = 0.0,
    ) -> None:
        """
        A highly generalised Ising model using the Metropolis algorithm for propagation.

        Both the nucleus spin and the model dimensionality can be arbitrarily defined,
        though the current implementation is specific to square hyperlattices.

        Args:
            size: Size of the simulated array
            dimensionality: Dimensionality of the model
            spin2: 2 ⋅ the nuclear spin
            spin_distribution: An array
            spin_dtype: dtype to use for the spins
            rng_seed: The seed to use for random state generation
        """
        self._rng_key = random.PRNGKey(seed=rng_seed)
        self.spin_states = jnp.asarray(
            spin_states or IsingModelND.get_spin_states(spin)
        )  # m_s
        self.state_shape = tuple([size] * dimensionality)
        self.spin_distribution = spin_distribution

        self.dimensionality = dimensionality
        self.size = size
        self.shape = tuple([self.size] * self.dimensionality)

        self.interaction_bilinear = interaction_bilinear  # J
        self.interaction_biquadratic = interaction_biquadratic  # K
        self.interaction_anisotropy = interaction_anisotropy  # D
        self.interaction_bicubic = interaction_bicubic  # L
        self.interaction_external_field = interaction_external_field  # H
        self.nuclear_magnetic_moment = nuclear_magnetic_moment  # μ

        # Generate a random initial state
        if initial_state is not None:
            assert spin_states is not None
            cast_ini_state = jnp.asarray(initial_state, dtype=TSpin)
            cast_spin_states = tuple(spin_states)
            self.initial_state = IsingStateND(cast_ini_state, cast_spin_states)
        else:
            self.initial_state = self.get_random_state()

        __nn_kernel = generate_binary_structure(self.dimensionality, 1)
        np.put(__nn_kernel, __nn_kernel.size // 2, False)
        self._nn_kernel = lists_to_tuples(__nn_kernel.tolist())

    @property
    def rng_key(self) -> ScalarInt:
        """
        Splits the RNG key and returns a fresh key.
        """
        self._rng_key, k = cast(tuple[ScalarInt, ScalarInt], random.split(self._rng_key))
        return k

    def get_random_state(self) -> IsingStateND:
        arr: TSpins = random.choice(
            self._rng_key,
            self.spin_states,
            self.state_shape,
            replace=True,
            p=self.spin_distribution,
        )

        return IsingStateND(arr, self.spin_states)

    @staticmethod
    def get_spin_states(spin: float) -> TSpins:
        spins: TSpins = jnp.arange(-spin, spin + 1, 1)

        return spins

    def get_energy(self, state: IsingStateND) -> float:
        """
        Calculates the Hamiltonian of the current state
        """
        H: float = get_hamiltonian(
            state.arr,
            self._nn_kernel,
            self.interaction_bilinear,
            self.interaction_biquadratic,
            self.interaction_anisotropy,
            self.interaction_bicubic,
            self.interaction_external_field,
            self.nuclear_magnetic_moment,
        )

        return H

    def get_magnetisation(self, state: IsingStateND) -> float:
        return get_magnetisation_density(state.arr, self.nuclear_magnetic_moment)

    def run_steps(
        self,
        states: list[IsingStateND],
        steps: int,
        temperature_or_temperatures: float | Sequence[float],
        bc_mode: TBCModes = "constant",
        bc_mode_value: float | None = 0.0,
    ) -> None:
        temperatures = jnp.array(
            (
                [temperature_or_temperatures]
                if isinstance(temperature_or_temperatures, float)
                else temperature_or_temperatures
            )
        )

        N = temperatures.size

        assert len(states) == N

        states_ = jnp.array([state.arr for state in states])
        rng_keys_ = random.split(self.rng_key, N)
        betas_ = 1 / (constants.Boltzmann * temperatures)

        devices = jax.devices()
        ND = len(devices)

        states__ = jnp.stack(jnp.split(states_, ND))
        rng_keys__ = jnp.stack(jnp.split(rng_keys_, ND))
        betas__ = jnp.stack(jnp.split(betas_, ND))

        end_states = pvrun_mcmc_steps(
            steps,
            rng_keys__,
            states__,
            states[0].spin_states,
            betas__,
            self.interaction_bilinear,
            self.interaction_biquadratic,
            self.interaction_anisotropy,
            self.interaction_bicubic,
            self.interaction_external_field,
            self.nuclear_magnetic_moment,
            bc_mode,
            bc_mode_value,
        )

        flattened_end_states = jnp.reshape(end_states, (-1, *end_states.shape[2:]))
        for i, end_state in enumerate(flattened_end_states):
            # Update in-place to skip excessive object construction
            states[i].arr = end_state

    def get_equilibrium_energies_and_magnetisations(
        self,
        states: list[IsingStateND],
        steps: int,
        temperature_or_temperatures: float | Sequence[float],
        bc_mode: TBCModes = "constant",
        bc_mode_value: float | None = 0.0,
    ):
        temperatures = jnp.array(
            (
                [temperature_or_temperatures]
                if isinstance(temperature_or_temperatures, float)
                else temperature_or_temperatures
            )
        )

        N = temperatures.size

        assert len(states) == N

        states_ = jnp.array([state.arr for state in states])
        rng_keys_ = jnp.repeat(random.split(self.rng_key, steps)[None, ...], N, axis=0)
        betas_ = temperature_to_beta(temperatures)

        devices = jax.devices()
        ND = len(devices)

        states__ = jnp.stack(jnp.split(states_, ND))
        rng_keys__ = jnp.stack(jnp.split(rng_keys_, ND))
        betas__ = jnp.stack(jnp.split(betas_, ND))

        energies_and_magnetisations = jnp.asarray(
            pvget_equilibrium_energy_and_magnetisation(
                states__,
                rng_keys__,
                states[0].spin_states,
                betas__,
                self._nn_kernel,
                self.interaction_bilinear,
                self.interaction_biquadratic,
                self.interaction_anisotropy,
                self.interaction_bicubic,
                self.interaction_external_field,
                self.nuclear_magnetic_moment,
                bc_mode,
                bc_mode_value,
            )
        )

        energies_and_magnetisations = jnp.moveaxis(energies_and_magnetisations, 0, -1)

        flattened_energies_and_magnetisations = jnp.reshape(
            energies_and_magnetisations, (-1, *energies_and_magnetisations.shape[2:])
        )

        return flattened_energies_and_magnetisations
