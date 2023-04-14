from enum import StrEnum, auto


class Algorithm(StrEnum):
    METROPOLIS_HASTINGS = auto()
    GLAUBER = auto()
    WOLFF = auto()
    SWENDSEN_WANG = auto()
