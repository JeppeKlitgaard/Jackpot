from enum import StrEnum, auto


class BCMode(StrEnum):
    CONSTANT = auto()
    PERIODIC = auto()


class Algorithm(StrEnum):
    METROPOLIS_HASTINGS = auto()
    WOLFF = auto()
    SWENDSEN_WANG = auto()
