from __future__ import annotations

import warnings
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, lax, random
from jax.random import KeyArray
from jaxtyping import Float
from matplotlib.figure import Figure

from ising.algorithms.cluster import wolff_sweep
from ising.algorithms.local import (
    glauber_step,
    glauber_sweep,
    metropolis_hastings_step,
    metropolis_hastings_sweep,
)
from ising.module import EnsamblableModule
from ising.primitives.measure import get_hamiltonian, get_magnetisation_density
from ising.types import Algorithm
from ising.typing import (
    RNGKey,
    ScalarInt,
    TEvolveAlgorithm,
    TShape,
    TSpin,
    TSpins,
)


class Environment(EnsamblableModule):
    """
    Dataclass that holds details about the environment of the model.

    Many properties are held static and would require recompilation if changed.
    """

    # Possible spin states
    spin_states: tuple[float, ...] = eqx.static_field()

    # Coldness
    beta: Float[Array, "dim"]

    # Interaction parameters
    interaction_bilinear: float = eqx.static_field()
    interaction_biquadratic: float = eqx.static_field()
    interaction_anisotropy: float = eqx.static_field()
    interaction_bicubic: float = eqx.static_field()
    interaction_external_field: float = eqx.static_field()

    # Nuclear magnetic moment
    nuclear_magnetic_moment: float = eqx.static_field()

    # Method
    algorithm: Algorithm = eqx.static_field()

    # Configuration flags
    # Whether to apply probablistic cluster acceptance modification
    # This is needed for systems with external fields or anisotropy
    # But the algorithm has a higher (non-zero) rejection rate in this
    # case, so when not needed it should be disabled
    probabilistic_cluster_accept: bool = eqx.static_field()

    def __post_init__(self) -> None:
        is_cluster_algorithm = self.algorithm in [
            Algorithm.SWENDSEN_WANG,
            Algorithm.WOLFF,
        ]

        has_external_interactions = any(
            self.interaction_external_field, self.interaction_anisotropy
        )

        if (
            is_cluster_algorithm
            and has_external_interactions
            and not self.probabilistic_cluster_accept
        ):
            _msg = (
                "System environment has external interactions and is using cluster "
                "algorithm, but has not enabled the probablistic cluster "
                "accept modification!"
            )
            warnings.warn(_msg, UserWarning)

    @classmethod
    @eqx.filter_jit
    def from_spin(
        cls,
        spin: TSpin,
        *,
        beta: float,
        interaction_bilinear: float,
        interaction_biquadratic: float,
        interaction_anisotropy: float,
        interaction_bicubic: float,
        interaction_external_field: float,
        nuclear_magnetic_moment: float,
        algorithm: Algorithm,
        probabilistic_cluster_accept: bool,
    ) -> Self:
        """
        Construct an environment from a half-integer spin given as a float.
        """
        spins = tuple(np.arange(-spin, spin + 1.0, 1.0))

        return cls(
            spin_states=spins,
            beta=beta,
            interaction_bilinear=interaction_bilinear,
            interaction_biquadratic=interaction_biquadratic,
            interaction_anisotropy=interaction_anisotropy,
            interaction_bicubic=interaction_bicubic,
            interaction_external_field=interaction_external_field,
            nuclear_magnetic_moment=nuclear_magnetic_moment,
            algorithm=algorithm,
            probabilistic_cluster_accept=probabilistic_cluster_accept,
        )


class Measurements(EnsamblableModule):
    """
    Represents one or more measurements of the system taken at a particular
    step for the state with the given id.
    """

    steps: ScalarInt
    state_id: ScalarInt
    spins_shape: TShape

    energy: Float[Array, "a"]
    magnetisation_density: Float[Array, "a"]

    @property
    def vectorisation_shape(self) -> TShape:
        return self.energy.shape[:-1]

    @property
    def sweeps(self) -> float:
        return self.steps / np.prod(self.spins_shape)


class State(EnsamblableModule):
    """
    Represents an (immutable) state of the system.
    """

    spins: TSpins
    dim: int = eqx.static_field()

    env: Environment

    id_: int
    steps: int = 0

    @property
    def shape(self) -> TShape:
        return self.spins.shape[-self.dim :]

    @property
    def vectorisation_shape(self) -> TShape:
        return self.spins.shape[: -self.dim]

    @property
    def sweeps(self) -> float:
        return self.steps / self.spins.size

    @classmethod
    @eqx.filter_jit
    def uniform_random_square_from_env(
        cls,
        *,
        rng_key: KeyArray,
        dimensions: int,
        size: int,
        env: Environment,
        id_: int,
    ) -> Self:
        """
        Construct a state from an environment and dimensional information.

        The initial state is constructed as a uniform random distribution of
        the available spin states.
        """
        shape = tuple([size] * dimensions)
        spins = random.choice(
            key=rng_key,
            a=np.asarray(env.spin_states),
            shape=shape,
            replace=True,
            p=None,  # Uniform,
        )

        return cls(spins=spins, dim=spins.ndim, env=env, id_=id_)

    @classmethod
    @eqx.filter_jit
    def minimum_square_from_env(
        cls,
        *,
        rng_key: KeyArray,
        dimensions: int,
        size: int,
        env: Environment,
        id_: int,
    ) -> Self:
        """
        Construct a state from an environment and dimensional information.

        The initial state is constructed entirely from the minimal spins states.
        """
        shape = tuple([size] * dimensions)
        spins = random.choice(
            key=rng_key,
            a=np.asarray(env.spin_states),
            shape=shape,
            replace=True,
            p=None,  # Uniform,
        )
        spin = np.min(env.spin_states)
        spins = spin * np.ones(shape)

        return cls(spins=spins, dim=spins.ndim, env=env, id_=id_)

    @eqx.filter_jit
    def evolve_steps(self, steps: int, rng_key: RNGKey) -> Self:
        """
        Evolves the state for `steps` steps using the algorithm given in
        `state.env`.

        Returns the evolved state.
        """
        if not steps:
            return self

        keys = random.split(rng_key, num=steps)

        evolver: TEvolveAlgorithm
        match self.env.algorithm:
            case Algorithm.METROPOLIS_HASTINGS:
                evolver = metropolis_hastings_step

            case Algorithm.GLAUBER:
                evolver = glauber_step

            case Algorithm.WOLFF:
                raise ValueError(
                    "Wolff is a cluster algorithm and can only do full sweeps. "
                    "Use `evolve_sweeps`"
                )

            case Algorithm.SWENDSEN_WANG:
                raise ValueError(
                    "Swendsen-Wang is a cluster algorithm and can only do "
                    "full sweeps. "
                    "Use `evolve_sweeps`"
                )

        # For loop that carries out individual steps
        def body_fun(i: int, state: State) -> State:
            out: State = evolver(
                keys[i],
                state,
            )
            return out

        self = lax.fori_loop(0, steps, body_fun, self)

        return self

    @eqx.filter_jit
    def evolve_sweeps(self, sweeps: int, rng_key: RNGKey) -> Self:
        """
        Evolves the state for `sweeps` sweeps using the algorithm given in
        `state.env`.

        Returns the evolved state.
        """
        if not sweeps:
            return self

        keys = random.split(rng_key, num=sweeps)

        evolver: TEvolveAlgorithm
        match self.env.algorithm:
            case Algorithm.METROPOLIS_HASTINGS:
                evolver = metropolis_hastings_sweep

            case Algorithm.GLAUBER:
                evolver = glauber_sweep

            case Algorithm.WOLFF:
                evolver = wolff_sweep

            case Algorithm.SWENDSEN_WANG:
                raise ValueError(
                    "Swendsen-Wang is a cluster algorithm and can only do "
                    "full sweeps. "
                    "Use `evolve_sweeps`"
                )

        # For loop that carries out individual steps
        def body_fun(i: int, state: State) -> State:
            out: State = evolver(
                keys[i],
                state,
            )
            return out

        self = lax.fori_loop(0, sweeps, body_fun, self)

        return self

    @eqx.filter_jit
    def calculate_energy(self) -> Array:
        """
        Returns the energy of the system.
        """
        return get_hamiltonian(self)

    @eqx.filter_jit
    def calculate_magnetisation_density(self) -> Array:
        """
        Returns the magnetisation density of the system.
        """
        return get_magnetisation_density(self)

    @staticmethod
    @eqx.filter_jit
    def _get_single_measurements(
        state: State, rng_key: RNGKey, sweeps: int = 1
    ) -> tuple[Array, Array, Array]:
        """
        Transformable function that returns a single set of physical
        measurements on the system.
        """
        state = state.evolve_sweeps(sweeps=sweeps, rng_key=rng_key)
        energy = state.calculate_energy()
        magnetisation_density = state.calculate_magnetisation_density()

        return (
            state.steps,
            energy,
            magnetisation_density,
        )

    @eqx.filter_jit
    def measure(
        self, *, rng_key: RNGKey, num: int = 1, sweeps: int = 1
    ) -> Measurements:
        """
        Returns a series of physical measurements taken on the system.

        `num` independent measurements are taken of the system after it has
        been evolved for `sweeps` evolution sweeps.

        The states used to derived the measurements are evolved independently
        from the same initial state.
        """
        keys = random.split(rng_key, num=num)

        measurer = eqx.filter_vmap(in_axes=(None, 0, None))(
            self._get_single_measurements
        )
        measurement_steps, energies, magnetisation_densities = measurer(
            self, keys, sweeps
        )
        state_ids = jnp.repeat(jnp.asarray(self.id_), num)

        measurements = Measurements(
            state_id=state_ids,
            steps=measurement_steps,
            spins_shape=self.spins.shape,
            energy=energies,
            magnetisation_density=magnetisation_densities,
        )

        return measurements

    def plot(self) -> Figure:
        assert not self.is_vectorised

        norm = mpl.colors.Normalize(
            vmin=min(self.env.spin_states), vmax=max(self.env.spin_states)
        )

        fig = plt.figure()
        fig.set_size_inches(4, 4)

        match self.dim:
            case 2:
                ax = fig.add_subplot()
                im = ax.imshow(self.spins, norm=norm)

            case 3:
                ax = fig.add_subplot(111, projection="3d")

                x, y, z = np.indices(self.shape)
                im = ax.scatter(
                    x.ravel(),
                    y.ravel(),
                    z.ravel(),
                    c=self.spins.ravel(),
                    norm=norm,
                    alpha=0.2,
                )

                ax.axis("off")

            case _:
                raise ValueError(f"State plotting not supported for {self.dim}D states")

        colors = [im.cmap(norm(spin_state)) for spin_state in self.env.spin_states]
        patches = [
            mpatches.Patch(color=colors[i], label=str(spin_state))
            for i, spin_state in enumerate(self.env.spin_states)
        ]

        plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2)

        info_lines = (
            f"$S = {self.spins.sum():.1f}$",
            f"$β = {self.env.beta:.3f}$",
        )
        info_line = "\n".join(info_lines)
        plt.title(f"Spin states\n{info_line}")

        return fig
