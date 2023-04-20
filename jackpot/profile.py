import time
from typing import Self


class Stopwatch:
    def __init__(self, start_time: float) -> None:
        self.start_time = start_time

    @classmethod
    def start(cls) -> Self:
        return cls(start_time=time.time())

    def time(self) -> float:
        return time.time() - self.start_time
