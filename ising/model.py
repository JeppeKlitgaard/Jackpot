from typing import Any, Sequence, cast
import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
from jax import random
from jax.scipy.signal import convolve
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from scipy.ndimage import generate_binary_structure
from scipy import constants
from ising.utils.plot import make_alpha
from ising.typing import TSpins, TSpin
from ising.primitives import (
    run_mcmc_steps,
    multi_run_mcmc_steps
)
from joblib import Parallel, delayed


class IsingModelND:
    def __init__(
        self,
        dimensionality: int,
        *,
        size: int = 1024,
        spin: float = 0.5,
        spin_states: TSpins | None = None,
        spin_distribution: TSpins | tuple[float, ...] | None = None,
        initial_state: TSpins | None = None,
        rng_seed: int = 0,
        interaction_bilinear: float = 1.0,
        interaction_biquadratic: float = 0.0,
        interaction_anisotropy: float = 0.0,
        interaction_bicubic: float = 0.0,
        interaction_external_field: float = 0.0,
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
        self.spin_states = spin_states or IsingModelND.get_spin_states(spin)  # m_s
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

        # Generate a random initial state
        if initial_state is not None:
            self.initial_state = jnp.asarray(initial_state, dtype=TSpin)
        else:
            self.initial_state = self.get_random_state()

        self._nn_kernel = jnp.asarray(
            generate_binary_structure(self.dimensionality, 1), dtype=jnp.bool_
        )
        self._nn_kernel = self._nn_kernel.at[self._nn_kernel.size // 2].set(False)

    @property
    def rng_key(self) -> int:
        """
        Splits the RNG key and returns a fresh key.
        """
        self._rng_key, k = random.split(self._rng_key)

        return cast(int, k)

    def get_random_state(self) -> TSpins:
        state: TSpins = random.choice(
            self._rng_key,
            self.spin_states,
            self.state_shape,
            replace=True,
            p=self.spin_distribution,
        )

        return state

    @staticmethod
    def get_spin_states(spin: float) -> TSpins:
        spins: TSpins = jnp.arange(-spin, spin + 1, 1)

        return spins

    def calculate_hamiltonian(self, state: TSpins | None = None) -> float:
        """
        Calculates the Hamiltonian of the current state
        """
        state = state if state is not None else self.state
        H: float = 0

        if any(
            (
                self.interaction_biquadratic,
                self.interaction_anisotropy,
                self.interaction_bicubic,
            )
        ):
            state_sq = np.square(state)

        # J - Calculate bilinear exchange energy (nearest neighbour)
        if self.interaction_bilinear:
            H -= (
                self.interaction_bilinear
                * (state * convolve(state, self._nn_kernel, mode="same")).sum()
            )

        # K - Calculate biquadratic exchange energy (nearest neighbour)
        if self.interaction_biquadratic:
            H -= (
                self.interaction_biquadratic
                * (state_sq * convolve(state_sq, self._nn_kernel, mode="same")).sum()
            )

        # D - Calculate anisotropy energy
        if self.interaction_anisotropy:
            H -= self.interaction_anisotropy * state_sq.sum()

        # L - Calculate bicubic exchange energy (nearest neighbour)
        if self.interaction_bicubic:
            H -= (
                self.interaction_bicubic
                * 2
                * (state * convolve(state_sq, self._nn_kernel, mode="same")).sum()
            )

        # H - Calculate external field energy
        if self.interaction_external_field:
            H -= self.interaction_external_field * state.sum()

        return H

    def run_steps(
        self,
        steps: int,
        temperature_or_temperatures: float | Sequence[float],
        initial_state_or_states: Sequence[TSpins] | TSpins | None = None,
    ) -> None:

        temperatures = jnp.array(
            [temperature_or_temperatures]
            if isinstance(temperature_or_temperatures, float)
            else temperature_or_temperatures
        )

        N = len(temperatures)
        tile_shape = (1, N)


        if initial_state_or_states is None:
            initial_states = self.initial_state[None, ...].repeat(N, axis=0)
        elif initial_state_or_states.ndim == self.dimensionality:
            initial_states = jnp.array(initial_state_or_states)[None, ...].repeat(N, axis=0)
        else:
            initial_states = jnp.array(initial_state_or_states)

        steps_tuple = jnp.array(steps).repeat(N)
        betas_tuple = jnp.array(list((1 / (constants.Boltzmann * T) for T in temperatures)))

        end_states = multi_run_mcmc_steps(
            steps,
            self.rng_key,
            initial_states,
            self.spin_states,
            betas_tuple,
            self.interaction_bilinear,
            self.interaction_biquadratic,
            self.interaction_anisotropy,
            self.interaction_bicubic,
            self.interaction_external_field,
        )

        return end_states

    # def _run_mcmc_step(self) -> None:
    #     idx = self._get_random_point_idx()
    #     spin_prop = random.choice(self.rng_key, self.spin_states)
    #     state_prop = jnp.asarray(self.state.copy())
    #     state_prop = state_prop.at[idx].set(spin_prop)

    #     # This can be done more efficiently since interaction length is only
    #     # to nearest neighbour
    #     # TODO
    #     # H_init = self.calculate_hamiltonian()
    #     # H_prop = self.calculate_hamiltonian(state_prop)

    #     # H_delta_old = H_prop - H_init

    #     H_delta = get_hamiltonian_delta(
    #         self.state,
    #         idx,
    #         spin_prop,
    #         self.interaction_bilinear,
    #         self.interaction_biquadratic,
    #         self.interaction_anisotropy,
    #         self.interaction_bicubic,
    #         self.interaction_external_field,
    #     )

    #     # Change if new energy is lower
    #     if H_delta < 0:
    #         self.state = state_prop

    #     # Otherwise change with weighted probability
    #     elif jnp.exp(-self.beta * H_delta) > random.uniform(self.rng_key):
    #         self.state = state_prop

    #     # Otherwise keep old state

    def plot_state(self, state: TSpins) -> Figure:
        norm = mpl.colors.Normalize(
            vmin=self.spin_states.min(), vmax=self.spin_states.max()
        )

        fig = plt.figure()
        fig.set_size_inches(12, 12)

        match self.dimensionality:
            case 2:
                ax = fig.add_subplot()
                im = ax.imshow(state, norm=norm)

            case 3:
                ax = fig.add_subplot(111, projection="3d")

                x, y, z = np.indices(state.shape)
                im = ax.scatter(
                    x.ravel(),
                    y.ravel(),
                    z.ravel(),
                    c=state.ravel(),
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

        info_line = f"$S = {state.sum():.1f}$"
        plt.title(f"Spin states\n{info_line}")

        return fig
