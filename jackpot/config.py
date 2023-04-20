from enum import StrEnum, auto

from pydantic import BaseSettings

from jackpot.types import Algorithm


class JaxPlatform(StrEnum):
    TPU = auto()
    GPU = auto()
    CPU = auto()


class Config(BaseSettings):
    # base
    jax_platform: JaxPlatform
    do_profiling: bool
    enable_64_bit: bool
    do_autocorrelation: bool
    plot_states: bool

    # experiment
    rng_seed: int | None
    spin: float
    dimensions: int
    size: int
    total_samples: int
    n_samples_vectorise: int
    loop_n_temps_y: bool

    # environment
    interaction_bilinear: float
    interaction_biquadratic: float
    interaction_anisotropy: float
    interaction_bicubic: float
    interaction_external_field: float
    nuclear_magnetic_moment: float

    # environment config
    algorithm: Algorithm
    probabilistic_cluster_accept: bool

    # temperatures
    temp_min: float
    temp_max: float
    n_temps: int

    # thermalisation
    thermalise_steps: int
    thermalise_sweeps_per_step: int

    # autocorrelation
    autocorrelation_steps: int
    autocorrelation_sweeps_per_step: int

    # measure
    measure_equilibrium_steps: int
    measure_equilibrium_num: int
    measure_equilibrium_sweeps_per_step: int
