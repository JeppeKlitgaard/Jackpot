import math
from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any

import numpy as np
from jax import lax
from jax.experimental import io_callback
from tqdm import tqdm as tqdm_default

from jackpot.typing import T


def _make_device_calls(
    num: int,
    known_total: int,
    print_rate: int,
    tqdm: type[tqdm_default],
    should_close: bool,
):
    """
    Makes the host and device functions needed to facilitate the progress bar
    and communication.

    Returns the device update function.
    """
    assert print_rate != 0
    assert print_rate < num

    pbar: None | tqdm_default = None
    lock = Lock()

    max_step = num - 1
    remainder = max_step % print_rate

    num_inits = 0  # Track how many parallel calls we get
    num_closes = 0

    def host_update_pbar(
        *, num_steps: int, initial: bool = False, close: bool = False
    ) -> None:
        """
        This is called on the host.

        Since we support parallel environments we need to use a lock.
        """
        nonlocal num_inits
        nonlocal num_closes
        nonlocal pbar

        # On my system locks are generally held for ≈ 5e-5 sec
        lock.acquire(timeout=0.1)

        if pbar is None:
            pbar = tqdm(total=max(1, known_total), leave=True)

        if initial:
            num_inits += 1
            new_total = num_inits * num_steps
            pbar.total = known_total or new_total

        pbar.update(num_steps)

        if close and should_close:
            num_closes += 1
            if num_closes >= num_inits:
                pbar.close()

        lock.release()

    def device_update_pbar(*, step: int) -> Callable[..., None]:
        """
        This is called on the device and must be JAX jittable
        """
        initial = step == 0
        lax.cond(
            initial,
            lambda: io_callback(
                host_update_pbar, None, ordered=False, num_steps=1, initial=True
            ),
            lambda: None,
        )

        update_full = (step % print_rate == 0) & ~initial
        lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            update_full,
            lambda: io_callback(
                host_update_pbar, None, ordered=False, num_steps=print_rate
            ),
            lambda: None,
        )

        update_rest = (step == num - remainder) & ~update_full
        lax.cond(
            # update tqdm by `remainder`
            update_rest,
            lambda: io_callback(
                host_update_pbar, None, ordered=False, num_steps=remainder, close=True
            ),
            lambda: None,
        )

    return device_update_pbar


def make_fori_loop(
    tqdm: type[tqdm_default] = tqdm_default,
    known_total: int = 0,
    print_rate: int = 100,
    num_prints: int | None = None,
    should_close: bool = False,
) -> Callable[..., tuple[Any, Sequence[Any]]]:
    """
    Returns a function that replaces `jax.lax.fori_loop`.

    This supports `pmap` transformations and all JAX accelerators.

    Parameters:
        tqdm: a tqdm function. This allows the user to specify,
            for example, tqdm.notebook.tqdm
        print_rate: Will print every `print_rate`th scan
        num_prints: Calculates the appropriate `print_rate` to get a total of
            `num_prints` prints.
    """

    def fori_loop(
        lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T
    ) -> T:
        nonlocal print_rate

        num = upper - lower
        assert num > 0

        if num_prints is not None:
            print_rate = math.ceil(num / num_prints)

        update_pbar = _make_device_calls(
            num=num,
            known_total=known_total,
            print_rate=print_rate,
            tqdm=tqdm,
            should_close=should_close,
        )

        def _body_fun(i: int, val: T) -> T:
            update_pbar(step=i)
            return body_fun(i, val)

        return lax.fori_loop(
            lower=lower,
            upper=upper,
            body_fun=_body_fun,
            init_val=init_val,
        )

    return fori_loop


def make_scan(
    tqdm: type[tqdm_default] = tqdm_default,
    known_total: int = 0,
    print_rate: int = 100,
    num_prints: int | None = None,
    should_close: bool = False,
) -> Callable[..., tuple[Any, Sequence[Any]]]:
    """
    Returns a function that replaces `jax.lax.scan`.

    This supports `pmap` transformations and all JAX accelerators.

    Parameters:
        tqdm: a tqdm function. This allows the user to specify,
            for example, tqdm.notebook.tqdm
        print_rate: Will print every `print_rate`th scan
        num_prints: Calculates the appropriate `print_rate` to get a total of
            `num_prints` prints.
    """

    def scan(
        f: Callable[[T, Any], tuple[T, Any]],
        init: T,
        xs: Sequence[Any] | None,
        length: int | None = None,
        reverse: bool = False,
        unroll: int = 1,
    ) -> tuple[Any, Sequence[Any]]:
        nonlocal print_rate

        if xs is None:
            if length is None:
                _msg = "Cannot have `xs` and `length` be `None`. Define one of them."
                raise ValueError(_msg)

            num = length
        else:
            num = len(xs)

        if num_prints is not None:
            print_rate = math.ceil(num / num_prints)

        update_pbar = _make_device_calls(
            num=num,
            known_total=known_total,
            print_rate=print_rate,
            tqdm=tqdm,
            should_close=should_close,
        )

        def body_fun(carry: T, x: Any) -> tuple[T, Any]:
            assert isinstance(x, tuple)
            step, x_inner = x

            update_pbar(step=step)
            return f(carry, x_inner)

        steps = np.arange(num)
        return lax.scan(
            f=body_fun,
            init=init,
            xs=(
                steps,
                xs,
            ),
            length=length,
            reverse=reverse,
            unroll=unroll,
        )

    return scan
