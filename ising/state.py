from typing import Self

import equinox as eqx
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, lax, random
from jax.random import KeyArray
from jaxtyping import Float, Int
from matplotlib.figure import Figure

from ising.algorithms import metropolis_hastings_move
from ising.module import EnsamblableModule
from ising.primitives2 import get_hamiltonian, get_magnetisation_density
from ising.types import Algorithm, BCMode
from ising.typing import RNGKey, TFloatParam, TShape, TSpin


class Environment(EnsamblableModule):
    """
    Dataclass that holds details about the environment of the model.

    Many properties are held static and would require recompilation if changed.
    """

    # Possible spin states
    spin_states: tuple[TSpin, ...] = eqx.static_field()

    # Coldness
    beta: Float[Array, "dim"]

    # Interaction parameters
    interaction_bilinear: TFloatParam = eqx.static_field()
    interaction_biquadratic: TFloatParam = eqx.static_field()
    interaction_anisotropy: TFloatParam = eqx.static_field()
    interaction_bicubic: TFloatParam = eqx.static_field()
    interaction_external_field: TFloatParam = eqx.static_field()

    # Nuclear magnetic moment
    nuclear_magnetic_moment: TFloatParam = eqx.static_field()

    # Method
    algorithm: Algorithm = eqx.static_field()

    # Boundary conditions
    bc_mode: BCMode = eqx.static_field()
    bc_mode_value: TSpin | None = eqx.static_field()

    def __post_init__(
        self,
    ):
        # Validate
        if self.bc_mode == "constant" and self.bc_mode_value is None:
            raise ValueError(
                "Can't have unset boundary condition value with bc_mode='constant'"
            )

    @classmethod
    @eqx.filter_jit
    def from_spin(
        cls,
        spin: TSpin,
        *,
        beta: TFloatParam,
        interaction_bilinear: TFloatParam,
        interaction_biquadratic: TFloatParam,
        interaction_anisotropy: TFloatParam,
        interaction_bicubic: TFloatParam,
        interaction_external_field: TFloatParam,
        nuclear_magnetic_moment: TFloatParam,
        algorithm: Algorithm,
        bc_mode: BCMode,
        bc_mode_value: TSpin | None = None,
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
            bc_mode=bc_mode,
            bc_mode_value=bc_mode_value,
        )


class Measurements(EnsamblableModule):
    """
    Represents one or more measurements of the system taken at a particular
    step for the state with the given id.
    """

    steps: Int[Array, "1"]
    state_id: Int[Array, "1"]

    energy: Float[Array, "a"]
    magnetisation_density: Float[Array, "a"]

    @property
    def vectorisation_shape(self) -> TShape:
        return self.energy.shape[:-1]


class State(EnsamblableModule):
    """
    Represents an (immutable) state of the system.
    """

    spins: Float[Array, "*dims"]
    dim: Int[Array, "1"] = eqx.static_field()

    env: Environment

    id_: Int[Array, "1"]
    steps: Int[Array, "1"] = 0

    @property
    def shape(self) -> TShape:
        return self.spins.shape[-self.dim:]

    @property
    def vectorisation_shape(self) -> TShape:
        return self.spins.shape[:-self.dim]

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
    def evolve(self, steps: int, rng_key: RNGKey) -> Self:
        """
        Evolves the state for `steps` steps using the algorithm given in
        `state.env`.

        Returns the evolved state.
        """
        if not steps:
            return self

        keys = random.split(rng_key, num=steps)

        evolver = None
        match self.env.algorithm:
            case Algorithm.METROPOLIS_HASTINGS:
                evolver = metropolis_hastings_move

            case Algorithm.WOLFF:
                evolver = None

            case Algorithm.SWENDSEN_WANG:
                evolver = None

        # For loop that carries out individual steps
        def body_fun(i: int, state: State) -> State:
            out: State = evolver(
                keys[i],
                state,
            )
            return out

        self = lax.fori_loop(0, steps, body_fun, self)

        # Update number of steps
        where = lambda _: _.steps
        self = eqx.tree_at(where, self, self.steps + steps)

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
        state: Self, rng_key: RNGKey, steps: int = 1
    ) -> tuple[Array, Array, Array]:
        """
        Transformable function that returns a single set of physical
        measurements on the system.
        """
        new_state = state.evolve(steps=steps, rng_key=rng_key)
        energy = new_state.calculate_energy()
        magnetisation_density = new_state.calculate_magnetisation_density()

        return (
            new_state.steps,
            energy,
            magnetisation_density,
        )

    @eqx.filter_jit
    def measure(
        self, *, rng_key: RNGKey, num: int = 1, steps: int = 1
    ) -> Measurements:
        """
        Returns a series of physical measurements taken on the system.

        `num` independent measurements are taken of the system after it has
        been evolved for `steps` evolution steps.

        The states used to derived the measurements are evolved independently
        from the same initial state.
        """
        keys = random.split(rng_key, num=num)

        measurer = eqx.filter_vmap(in_axes=(None, 0, None))(
            self._get_single_measurements
        )
        measurement_steps, energies, magnetisation_densities = measurer(
            self, keys, steps
        )

        measurements = Measurements(
            state_id=self.id_,
            steps=measurement_steps,
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

        info_line = f"$S = {self.spins.sum():.1f}$"
        plt.title(f"Spin states\n{info_line}")

        return fig
