from jax import Array
from jaxtyping import Float
from scipy import constants


def beta_to_temp(beta: Float[Array, "*dim"]) -> Float[Array, "*dim"]:
    reciprocal: Float[Array, "*dim"] = constants.Boltzmann * beta
    return 1.0 / reciprocal

def temp_to_beta(temp: Float[Array, "*dim"]) -> Float[Array, "*dim"]:
    reciprocal: Float[Array, "*dim"] = constants.Boltzmann * temp
    return 1.0 / reciprocal