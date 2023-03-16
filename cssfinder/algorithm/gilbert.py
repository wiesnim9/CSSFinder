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


"""Module contains public interface of Gilbert algorithm."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING, Iterable

from cssfinder.algorithm import backend as _backend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from cssfinder.cssfproject import AlgoMode, Backend, Precision


class Gilbert:
    """Class interface of gilbert algorithm."""

    def __init__(
        self,
        *,
        initial: npt.NDArray[np.complex128],
        depth: int,
        quantity: int,
        mode: AlgoMode,
        backend: Backend,
        precision: Precision,
        visibility: float,
        is_debug: bool = False,
    ) -> None:
        self.initial = initial
        self.depth = depth
        self.quantity = quantity

        self.mode = mode
        self.precision = precision
        self.visibility = visibility

        self.is_debug = is_debug

        backend_type = _backend.select(backend, self.precision)
        self.backend = backend_type(
            self.initial,
            self.depth,
            self.quantity,
            self.mode,
            self.visibility,
            is_debug=self.is_debug,
        )

    def set_symmetries(
        self, symmetries: list[list[npt.NDArray[np.complex128]]]
    ) -> None:
        """Set symmetries to use during calculations.

        This operation may involve type conversion and copying of symmetries, therefore
        if may be slow and should should be done only once.

        Parameters
        ----------
        symmetries : list[list[npt.NDArray[np.complex128]]]
            Array of symmetries.

        """
        self.backend.set_symmetries(symmetries)

    def set_projection(self, projection: npt.NDArray[np.complex128]) -> None:
        """Set projection to use during calculations.

        This operation may involve type conversion and copying of symmetries, therefore
        if may be slow and should should be done only once.

        Parameters
        ----------
        projection : npt.NDArray[np.complex128]
            Projection matrix.

        """
        self.backend.set_projection(projection)

    def run(
        self,
        max_epochs: int,
        iterations_per_epoch: int,
        max_corrections: int,
    ) -> Iterable[int]:
        """Run a specified number of epochs with a given number of iterations per epoch
        or until a certain amount of corrections is found.

        Parameters
        ----------
        max_epochs : int
            The number of epochs to run.
        iterations_per_epoch : int
            The number of iterations to run per epoch. Within epoch, no stop conditions
            are checked, therefore number of corrections can exceed expected threshold.
        max_corrections : int
            The maximum number of corrections allowed.

        Yields
        ------
        int
            An iterator over the epoch indices, up to the point where the maximum
            number of expected corrections is reached.

        Notes
        -----
        This method runs a certain number of epochs with a given number of iterations
        per epoch on a backend. It will yield the epoch index at each epoch iteration
        while running, and will stop if and when the number of corrections found by the
        backend exceeds the specified `max_corrections` or after performing number of
        epochs specified with `max_epochs`.

        """
        start = perf_counter()
        total_iterations = max_epochs * iterations_per_epoch
        epoch_index = 0

        for epoch_index in range(max_epochs):
            # Run N iterations of algorithm without checking stop conditions.
            self.backend.run_epoch(iterations_per_epoch, epoch_index)

            iterations_executed = (epoch_index + 1) * iterations_per_epoch
            logging.debug(
                "Executed %r iterations, total %r / %r (%.2f%%)",
                iterations_per_epoch,
                iterations_executed,
                total_iterations,
                (iterations_executed / total_iterations) * 100,
            )
            # Check if we already reached expected number of corrections
            if self.backend.get_corrections_count() >= max_corrections:
                logging.info(
                    "Reached expected maximal number of corrections %r",
                    max_corrections,
                )
                break

            yield epoch_index

        end = perf_counter()
        logging.info("Elapsed time: %r.", end - start)

        # Possibly trigger user defined code before ending execution.
        yield epoch_index
        return

    def get_state(self) -> npt.NDArray[np.complex128]:
        """Return current system state with all optimizations applied."""
        return self.backend.get_state()

    def get_corrections(self) -> list[tuple[int, int, float]]:
        """Return list of all corrections found during optimization."""
        return self.backend.get_corrections()

    def get_corrections_count(self) -> int:
        """Return number of all corrections found during optimization."""
        return self.backend.get_corrections_count()


class AlgorithmError(Exception):
    """Base for exceptions raised by gilbert algorithm."""


class AlgorithmNotSaturatedError(Exception):
    """Raised when action was performed on which required algorithm to finish execution
    on instance which was not run.
    """


class HookError(Exception):
    """Base class for hook error wrappers."""


class SaveStateHookError(HookError):
    """Wrapper for exceptions raised by save_state_hook."""


class SaveCorrectionsHookError(HookError):
    """Wrapper for exceptions raised by save_state_hook."""
