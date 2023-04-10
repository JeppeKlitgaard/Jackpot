import jax_dataclasses as jdc
import jax.numpy as jnp
from ising.typing import TSpin, TFloatParam, TBCMode, TMethod, RNGKey
from jax.random import KeyArray
from jax import random
from typing import Self
import numpy as np
from dataclasses import dataclass


@jdc.pytree_dataclass
class Environment:
    # Possible spin states
    spin_states: jdc.Static[tuple[TSpin, ...]]

    # Coldness
    beta: jdc.Static[TFloatParam]

    # Interaction parameters
    interaction_bilinear: jdc.Static[TFloatParam]
    interaction_biquadratic: jdc.Static[TFloatParam]
    interaction_anisotropy: jdc.Static[TFloatParam]
    interaction_bicubic: jdc.Static[TFloatParam]
    interaction_external_field: jdc.Static[TFloatParam]

    # Nuclear magnetic moment
    nuclear_magnetic_moment: jdc.Static[TFloatParam]

    # Method
    method: jdc.Static[TMethod]

    # Boundary conditions
    bc_mode: jdc.Static[TBCMode]
    bc_mode_value: jdc.Static[TSpin | None]

    def __post_init__(
        self,
    ):
        # Validate
        if self.bc_mode == "constant" and self.bc_mode_value is None:
            raise ValueError(
                "Can't have unset boundary condition value with bc_mode='constant'"
            )

    @classmethod
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
        method: TMethod,
        bc_mode: TBCMode,
        bc_mode_value: TSpin | None = None,
    ) -> Self:
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
            method=method,
            bc_mode=bc_mode,
            bc_mode_value=bc_mode_value,
        )

@jdc.pytree_dataclass
class State:
    spins: jnp.ndarray
    rng_key: RNGKey

    env: jdc.Static[Environment]

    # def rng_key(self) -> RNGKey:
    #     return self.rng_keys(num=1)

    # def rng_keys(self, num: int) -> tuple[RNGKey, ...]:
    #     self.rng_key, *k = random.split(self.rng_key, num=num + 1)
    #     return k

    # @property
    # def dim(self) -> int:
    #     return self.spins.ndim

    # @property
    # def shape(self) -> tuple[int, ...]:
    #     return self.spins.shape

    @classmethod
    def uniform_random_square_from_env(
        cls, *, rng_key: KeyArray, dimensions: int, size: int, env: Environment
    ) -> Self:
        spins_key, rng_key = random.split(rng_key, num=2)
        shape = tuple([size] * dimensions)
        spins = random.choice(
            key=spins_key,
            a=np.asarray(env.spin_states),
            shape=shape,
            replace=True,
            p=None,  # Uniform,
        )

        return cls(
            spins=spins,
            rng_key=rng_key,
            env=env,
        )
