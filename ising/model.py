from typing import Any
import numpy as np
import jax.numpy as jnp
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from scipy.ndimage import generate_binary_structure, convolve
from scipy import constants
from ising.utils.plot import make_alpha

SPIN_DT = np.float64
TSpin = np.float64
TSpins = npt.NDArray[TSpin]


class IsingModelND:
    def __init__(
        self,
        dimensionality: int,
        temperature: float,
        *,
        size: int = 1024,
        spin: float = 0.5,
        spin_states: TSpins | None = None,
        spin_distribution: TSpins | tuple[float] | None = None,
        initial_state: TSpins | None = None,
        rng_seed: int | None = None,
        interaction_bilinear: float = 1,
        interaction_biquadratic: float = 0,
        interaction_anisotropy: float = 0,
        interaction_bicubic: float = 0,
        interaction_external_field: float = 0,
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
        self._rng = np.random.default_rng(seed=rng_seed)
        self.spin_states = spin_states or np.arange(-spin, spin + 1, 1)  # m_s
        self.state_shape = tuple([size] * dimensionality)
        self.spin_distribution = spin_distribution

        self.beta = 1 / (constants.Boltzmann * temperature)

        self.interaction_bilinear = interaction_bilinear  # J
        self.interaction_biquadratic = interaction_biquadratic  # K
        self.interaction_anisotropy = interaction_anisotropy  # D
        self.interaction_bicubic = interaction_bicubic  # L
        self.interaction_external_field = interaction_external_field  # H

        # Generate a random initial state
        if initial_state is not None:
            self.state = initial_state
        else:
            self.set_random_state()

        self._nn_kernel = generate_binary_structure(self.dimensionality, 1)
        self._nn_kernel.put(self._nn_kernel.size // 2, False)

    def set_random_state(self) -> None:
        self.state = self._rng.choice(
            self.spin_states, self.state_shape, replace=True, p=self.spin_distribution
        )

    @staticmethod
    def get_spin_states(spin: float) -> TSpins:
        spins: TSpins = np.arange(-spin, spin, 1)

        return spins

    @property
    def size(self) -> int:
        size: int = self.state.shape[0]
        return size

    @property
    def dimensionality(self) -> int:
        return self.state.ndim

    @property
    def temperature(self) -> float:
        temp: float = 1 / (self.beta * constants.Boltzmann)
        return temp

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
                * (
                    state
                    * convolve(state, self._nn_kernel, mode="constant", cval=0)
                ).sum()
            )

        # K - Calculate biquadratic exchange energy (nearest neighbour)
        if self.interaction_biquadratic:
            H -= (
                self.interaction_biquadratic
                * (
                    state_sq
                    * convolve(state_sq, self._nn_kernel, mode="constant", cval=0)
                ).sum()
            )

        # D - Calculate anisotropy energy
        if self.interaction_anisotropy:
            H -= self.interaction_anisotropy * state_sq.sum()

        # L - Calculate bicubic exchange energy (nearest neighbour)
        if self.interaction_bicubic:
            H -= (
                self.interaction_bicubic
                * 2
                * (
                    state
                    * convolve(state_sq, self._nn_kernel, mode="constant", cval=0)
                ).sum()
            )

        # H - Calculate external field energy
        if self.interaction_external_field:
            H -= self.interaction_external_field * state.sum()

        return H

    def _get_random_point_idx(self) -> tuple[int, ...]:
        return tuple([np.random.randint(self.size) for n in range(self.dimensionality)])

    def _run_mcmc_step(self) -> None:
        idx = self._get_random_point_idx()
        spin_prop = np.random.choice(self.spin_states)
        state_prop = self.state.copy()
        state_prop[idx] = spin_prop

        # This can be done more efficiently since interaction length is only
        # to nearest neighbour
        # TODO
        H_init = self.calculate_hamiltonian()
        H_prop = self.calculate_hamiltonian(state_prop)

        H_delta = H_prop - H_init

        # Change if new energy is lower
        if H_delta < 0:
            self.state = state_prop

        # Otherwise change with weighted probability
        elif np.exp(-self.beta * H_delta) > self._rng.random():
            self.state = state_prop

        # Otherwise keep old state

    def plot_state(self, state: TSpins | None = None) -> Figure:
        state = state if state is not None else self.state

        norm = mpl.colors.Normalize(vmin=self.spin_states.min(), vmax=self.spin_states.max())

        fig = plt.figure()
        fig.set_size_inches(8, 8)

        match self.dimensionality:
            case 2:
                ax = fig.add_subplot()
                im = ax.imshow(state)

            case 3:
                ax = fig.add_subplot(111, projection="3d")

                x, y, z = np.indices(state.shape)
                im = ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=state.ravel())

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
