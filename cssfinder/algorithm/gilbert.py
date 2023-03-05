# Copyright 2023 Krzysztof Wiśniewski <argmaster.world@gmail.com>
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the “Software”), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""This module contains public interface of Gilbert algorithm."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from cssfinder.algorithm import backend as _backend
from cssfinder.cssfproject import AlgoMode, Backend, Precision
from cssfinder.io.asset_loader import State


class Gilbert:
    """Class interface of gilbert algorithm."""

    def __init__(
        self,
        initial: State,
        mode: AlgoMode,
        backend: Backend,
        precision: Precision,
        visibility: float,
        *,
        is_debug: bool = False,
    ) -> None:
        self.initial = initial
        self.mode = mode
        self.precision = precision
        self.visibility = visibility
        self.is_debug = is_debug

        backend_type = _backend.new(backend, self.precision)
        self.backend = backend_type(
            self.initial, self.mode, self.visibility, is_debug=self.is_debug
        )

        self._state: Optional[npt.NDArray[np.complex128]] = None
        self._corrections: Optional[list[tuple[int, int, float]]] = None

    def run(
        self,
        epochs: int,
        iterations: int,
        max_corrections: int,
        *,
        save_state_hook: Callable[[npt.NDArray[np.complex128]], None],
        save_corrections_hook: Callable[[list[tuple[int, int, float]]], None],
    ) -> None:
        """Run epochs of iterations each, or up to max_corrections found."""
        start = perf_counter()
        total_iterations = epochs * iterations

        for epoch_index in range(epochs):
            logging.info(
                "Executing epoch %r / %r (%.2f) - corrections: %r best: %r",
                epoch_index + 1,
                epochs,
                ((epoch_index + 1) / epochs) * 100,
                self.backend.corrections_count,
                self.backend.corrections[-1][2]
                if self.backend.corrections_count > 0
                else None,
            )
            # Run N iterations of algorithm without checking stop conditions.
            self.backend.run_epoch(iterations, epoch_index)

            self._state = self.backend.state
            self._corrections = self.backend.corrections

            try:
                save_state_hook(self._state)
            except Exception as exc:
                logging.critical("Exception occurred within save_state_hook() call.")
                raise SaveStateHookError() from exc

            try:
                save_corrections_hook(self._corrections)
            except Exception as exc:
                logging.critical(
                    "Exception occurred within save_corrections_hook() call."
                )
                raise SaveCorrectionsHookError() from exc

            iterations_executed = (epoch_index + 1) * iterations
            logging.debug(
                "Executed %r iterations, total %r / %r (%.2f)",
                iterations,
                iterations_executed,
                total_iterations,
                (iterations_executed / total_iterations) * 100,
            )
            # Check if we already reached expected number of corrections
            if self.backend.corrections_count >= max_corrections:
                logging.info(
                    "Reached expected maximal number of corrections %r", max_corrections
                )
                break

        end = perf_counter()
        logging.info("Elapsed time: %r.", end - start)

    @property
    def state(self) -> npt.NDArray[np.complex128]:
        """Returns correction from saturated algorithm."""
        if self._state is None:
            raise AlgorithmNotSaturatedError("Run algorithm first, to obtain state!")
        return self._state

    @property
    def corrections(self) -> list[tuple[int, int, float]]:
        """Returns correction from saturated algorithm."""
        if self._corrections is None:
            raise AlgorithmNotSaturatedError(
                "Run algorithm first to obtain corrections!"
            )
        return self._corrections


class AlgorithmError(Exception):
    """Base for exceptions raised by gilbert algorithm."""


class AlgorithmNotSaturatedError(Exception):
    """Raised when action was performed on which required algorithm to finish execution
    on instance which was not run."""


class HookError(Exception):
    """Base class for hook error wrappers."""


class SaveStateHookError(HookError):
    """Wrapper for exceptions raised by save_state_hook."""


class SaveCorrectionsHookError(HookError):
    """Wrapper for exceptions raised by save_state_hook."""
