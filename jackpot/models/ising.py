"""
Implementation of a generalised Ising Model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy.ndimage import generate_binary_structure

from jackpot.models.base import Model
from jackpot.primitives.convolve import convolve_with_wrapping
from jackpot.primitives.local import get_nearest_neighbours
from jackpot.typing import TSpin

if TYPE_CHECKING:
    from jackpot.state import State


class IsingModel(Model):
    """
    General Ising Model with a configurable interaction Hamiltonian.

    This is written in such a way that it will accommodate more than two
    possible spin states, though this does not necessarily lead to physically
    sensible outcomes. For a higher spin model see the Vector Potts Model
    implementation in `jackpot.models.vector_potts`.

    For explanation of the different interaction coefficients:
        See: http://arxiv.org/abs/2007.08593

    """

    # Interaction parameters
    interaction_bilinear: float = eqx.static_field()
    interaction_biquadratic: float = eqx.static_field()
    interaction_anisotropy: float = eqx.static_field()
    interaction_bicubic: float = eqx.static_field()
    interaction_external_field: float = eqx.static_field()

    # Nuclear magnetic moment
    nuclear_magnetic_moment: float = eqx.static_field()

    @classmethod
    def new(
        cls,
        spin: float,
        interaction_bilinear: float,
        interaction_biquadratic: float,
        interaction_anisotropy: float,
        interaction_bicubic: float,
        interaction_external_field: float,
        nuclear_magnetic_moment: float,
    ) -> Self:
        spin_states = tuple(np.arange(-spin, spin + 1.0, 1.0))

        return cls(
            spin_states=spin_states,
            interaction_bilinear=interaction_bilinear,
            interaction_biquadratic=interaction_biquadratic,
            interaction_anisotropy=interaction_anisotropy,
            interaction_bicubic=interaction_bicubic,
            interaction_external_field=interaction_external_field,
            nuclear_magnetic_moment=nuclear_magnetic_moment,
        )

    def get_magnetisation_density(self, state: State) -> Float[Array, ""]:
        magnetisation = self.nuclear_magnetic_moment * jnp.sum(state.spins)
        return magnetisation / jnp.size(state.spins)

    def get_hamiltonian(self, state: State) -> Float[Array, ""]:
        spins = state.spins

        # Find a kernel we can use with convolution
        binary_kernel = generate_binary_structure(state.dim, 1)
        np.put(binary_kernel, binary_kernel.size // 2, False)
        kernel = binary_kernel.astype(spins.dtype)

        H: Float[Array, ""] = jnp.asarray(0.0)
        spins_sq = spins**2

        spins_convolved = convolve_with_wrapping(spins, kernel=kernel)
        spins_sq_convolved = convolve_with_wrapping(spins_sq, kernel=kernel)

        # J - Calculate bilinear exchange energy (nearest neighbour)
        H -= self.interaction_bilinear * (spins * spins_convolved).sum()

        # K - Calculate biquadratic exchange energy (nearest neighbour)
        H -= self.interaction_biquadratic * (spins_sq * spins_sq_convolved).sum()

        # D - Calculate anisotropy energy
        H -= self.interaction_anisotropy * spins_sq.sum()

        # L - Calculate bicubic exchange energy (nearest neighbour)
        H -= self.interaction_bicubic * (
            (spins_sq * spins_convolved + spins * spins_sq_convolved).sum()
        )

        # H - Calculate external field energy
        H -= (
            self.nuclear_magnetic_moment * self.interaction_external_field * spins.sum()
        )

        return H

    def get_hamiltonian_delta(
        self,
        state: State,
        idx: Array,
        trial_spin: TSpin,
    ) -> Float[Array, ""]:
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
        H: Float[Array, ""] = jnp.asarray(0.0)

        current_spin = state.spins[tuple(idx)]
        delta_spin = trial_spin - current_spin

        neighbours = get_nearest_neighbours(
            state=state,
            idx=idx,
        )

        neighbours_sq = jnp.square(neighbours)
        delta_spin_sq = jnp.square(delta_spin)

        # J - Calculate bilinear exchange energy (nearest neighbour)
        H -= 2 * self.interaction_bilinear * (delta_spin * neighbours).sum()

        # K - Calculate biquadratic exchange energy (nearest neighbour)
        H -= 2 * self.interaction_biquadratic * (delta_spin_sq * neighbours_sq).sum()

        # D - Calculate anisotropy energy
        H -= self.interaction_anisotropy * delta_spin_sq

        # L - Calculate bicubic exchange energy (nearest neighbour)
        H -= (
            self.interaction_bicubic
            * (delta_spin_sq * neighbours + delta_spin * neighbours_sq).sum()
        )

        # H - Calculate external field energy
        H -= self.nuclear_magnetic_moment * self.interaction_external_field * delta_spin

        return H

    def get_cluster_linkage_factors(
        self,
        state: State,
        spins: Float[Array, "*dims"],
        neighbours: Float[Array, "ndim *dims"],
    ) -> Float[Array, "ndim *dims"]:
        """
        Calculates the linkage factors between neighbours and returns a linkage
        factor ND-array. This is used to determine whether to build a link
        between neighbours by cluster algorithms.

        Not that interaction coefficients are JIT compile-static and thus any
        Hamiltonian contributions with interaction coefficients that are zero
        are automatically discarded during JIT tree-shaking.

        Introduction of if statements to do manual tree-shaking would at best
        slow down tracing process and at worst lead to slower runtimes.

        See: http://arxiv.org/abs/2007.08593
        """
        # i = spins
        # j = neighbours
        link_factors = jnp.zeros_like(neighbours)

        # J - bilinear exchange energy (nearest neighbour)
        link_factors += self.interaction_bilinear * spins * neighbours

        # K - Calculate biquadratic exchange energy (nearest neighbour)
        link_factors += self.interaction_biquadratic * (spins**2) * (neighbours**2)

        # L - Calculate bicubic exchange energy (nearest neighbour)
        link_factors += self.interaction_bicubic * (
            (spins**2) * neighbours + spins * (neighbours**2)
        )

        # Note: Here we have another factor of 2^2 compared to most papers
        # This is again due to using ±0.5 for spins rather than ±1 and
        # different conventions on double counting
        link_factors *= -4.0
        link_factors *= state.beta

        # In case we're in a high spin system we need to nuke the ones that are
        # partially aligned
        is_same = spins == neighbours
        link_factors *= is_same

        return link_factors
