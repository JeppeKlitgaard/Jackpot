"""
Submodule holding different local and cluster lattice algorithms.
"""

from enum import StrEnum, auto

from jackpot.algorithms.base import Algorithm
from jackpot.algorithms.glauber import GlauberAlgorithm
from jackpot.algorithms.metropolis_hastings import MetropolisHastingsAlgorithm
from jackpot.algorithms.swendsen_wang import SwendsenWangAlgorithm
from jackpot.algorithms.wolff import WolffAlgorithm


class AlgorithmChoice(StrEnum):
    METROPOLIS_HASTINGS = auto()
    GLAUBER = auto()
    WOLFF = auto()
    SWENDSEN_WANG = auto()

    def resolve(self) -> type[Algorithm]:
        """
        We use AlgorithmChoice as a `StrEnum` to be able to configure our
        configuration object using plain text files (YAML), but need
        to resolve these into actual algorithm classes
        """
        match self.value:
            case AlgorithmChoice.METROPOLIS_HASTINGS:
                return MetropolisHastingsAlgorithm

            case AlgorithmChoice.GLAUBER:
                return GlauberAlgorithm

            case AlgorithmChoice.WOLFF:
                return WolffAlgorithm

            case AlgorithmChoice.SWENDSEN_WANG:
                return SwendsenWangAlgorithm

            case _:  # Makes mypy shut up
                raise ValueError("Invalid algorithm choice.")
