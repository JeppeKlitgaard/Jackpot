from __future__ import annotations

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

from jackpot.algorithms.base import Algorithm
from jackpot.models.base import Model
from jackpot.module import EnsamblableModule
from jackpot.modules import Measurement
from jackpot.typing import (
    RNGKey,
    TShape,
    TSpins,
)


class State(EnsamblableModule):
    """
    Represents an (immutable) state of the system.

    This object should contain all information about the system,
    how to measure the system and how to evolve the state.
    """

    spins: TSpins
    beta: Float[Array, ""]

    model: Model
    algorithm: Algorithm

    id_: int
    steps: int = 0
    sweeps: int = 0

    @property
    def shape(self) -> TShape:
        """
        The shape of the lattice.
        """
        return self.spins.shape[-self.dim :]

    @property
    def size(self) -> int:
        """
        The number of lattice sites.
        """
        return np.prod(self.shape)

    @property
    def dim(self) -> int:
        """
        The dimensionality of lattice.
        """
        return self.spins.ndim - len(self.transformation_shape)

    @property
    def transformation_shape(self) -> TShape:
        """
        The shape of the ensamble due to JAX transformations.
        """
        return jnp.asarray(self.beta).shape

    @classmethod
    @eqx.filter_jit
    def new_uniform_random(
        cls,
        *,
        rng_key: KeyArray,
        dimensions: int,
        size: int,
        beta: float,
        model: Model,
        algorithm: Algorithm,
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
            a=np.asarray(model.spin_states),
            shape=shape,
            replace=True,
            p=None,  # Uniform,
        )

        return cls(
            spins=spins,
            beta=beta,
            model=model,
            algorithm=algorithm,
            id_=id_,
            steps=0,
            sweeps=0,
        )

    @classmethod
    @eqx.filter_jit
    def new_minimal(
        cls,
        *,
        rng_key: KeyArray,
        dimensions: int,
        size: int,
        beta: float,
        model: Model,
        algorithm: Algorithm,
        id_: int,
    ) -> Self:
        """
        Construct a state from an environment and dimensional information.

        The initial state is constructed entirely from the minimal spins states.

        This may be handy in cases where there is no external field thus leading
        to bifurcation at low temperatures.
        """
        shape = tuple([size] * dimensions)
        spins = random.choice(
            key=rng_key,
            a=np.asarray(model.spin_states),
            shape=shape,
            replace=True,
            p=None,  # Uniform,
        )
        spin = np.min(model.spin_states)
        spins = spin * np.ones(shape)

        return cls(
            spins=spins,
            beta=beta,
            model=model,
            algorithm=algorithm,
            id_=id_,
            steps=0,
            sweeps=0,
        )

    @eqx.filter_jit
    def evolve_steps(self, rng_key: RNGKey, steps: int) -> Self:
        """
        Evolves the state for `steps` steps using the algorithm given in
        `state.algorithm`.

        Returns the evolved state.
        """
        if not steps:
            return self

        keys = random.split(rng_key, num=steps)

        # For loop that carries out individual steps
        def body_fun(i: int, state: State) -> State:
            out: State = state.algorithm.step(
                keys[i],
                state,
            )
            return out

        self = lax.fori_loop(0, steps, body_fun, self)

        return self

    @eqx.filter_jit
    def evolve_sweeps(self, rng_key: RNGKey, sweeps: int) -> Self:
        """
        Evolves the state for `sweeps` sweeps using the algorithm given in
        `state.algorithm`.

        Returns the evolved state.
        """
        if not sweeps:
            return self

        keys = random.split(rng_key, num=sweeps)

        # For loop that carries out individual steps
        def body_fun(i: int, state: State) -> State:
            out: State = state.algorithm.sweep(
                keys[i],
                state,
            )
            return out

        self = lax.fori_loop(0, sweeps, body_fun, self)

        # Update sweeps
        where = lambda s: s.sweeps
        self = eqx.tree_at(where, self, self.sweeps + sweeps)

        return self

    @eqx.filter_jit
    def calculate_energy(self) -> Array:
        """
        Returns the energy of the system.
        """
        assert not self.is_transformed
        return self.model.get_hamiltonian(self)

    @eqx.filter_jit
    def calculate_magnetisation_density(self) -> Array:
        """
        Returns the magnetisation density of the system.
        """
        assert not self.is_transformed
        return self.model.get_magnetisation_density(self)

    @eqx.filter_jit
    def measure(self) -> Measurement:
        """
        Returns a series of physical measurements taken on the system.

        `num` independent measurements are taken of the system after it has
        been evolved for `sweeps` evolution sweeps.

        The states used to derived the measurements are evolved independently
        from the same initial state.
        """
        assert not self.is_transformed
        energy = self.calculate_energy()
        magnetisation_density = self.calculate_magnetisation_density()

        return Measurement(
            state_id=self.id_,
            beta=self.beta,
            steps=self.steps,
            sweeps=self.sweeps,
            energy=energy,
            magnetisation_density=magnetisation_density,
        )

    @eqx.filter_jit
    def evolve_and_measure_multiple(
        self, *, rng_key: RNGKey, num: int = 1, sweeps: int = 1
    ) -> Measurement:
        """
        Returns a series of physical measurements taken on the system.

        `num` independent measurements are taken of the system after it has
        been evolved for `sweeps` evolution sweeps.

        The states used to derived the measurements are evolved independently
        from the same initial state.
        """

        @eqx.filter_vmap(in_axes=(0, None, None))
        def evolve_and_measure(
            rng_key: RNGKey, state: State, sweeps: int
        ) -> Measurement:
            state = state.evolve_sweeps(rng_key=rng_key, sweeps=sweeps)
            measurement = state.measure()

            return measurement

        keys = random.split(rng_key, num=num)
        measurements = evolve_and_measure(keys, self, sweeps)

        return measurements

    def plot(self, title: str | None = None) -> Figure:
        """
        Plots the state assuming it is in 2D or 3D.
        """
        assert not self.is_transformed

        norm = mpl.colors.Normalize(
            vmin=min(self.model.spin_states), vmax=max(self.model.spin_states)
        )

        fig = plt.figure()
        fig.set_size_inches(4, 4)

        match self.dim:
            case 2:
                ax = fig.add_subplot()
                im = ax.imshow(
                    self.spins, norm=norm, interpolation=None, rasterized=True
                )

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

        colors = [im.cmap(norm(spin_state)) for spin_state in self.model.spin_states]
        patches = [
            mpatches.Patch(color=colors[i], label=str(spin_state))
            for i, spin_state in enumerate(self.model.spin_states)
        ]

        plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, framealpha=0.2)

        info_lines = (
            f"$S = {self.spins.sum():.1f}$",
            f"$β = {self.beta:.3f}$",
        )
        info_line = "\n".join(info_lines)
        title = title or "Spin states"
        plt.title(f"{title}\n{info_line}")

        return fig
