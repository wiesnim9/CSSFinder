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

from cssfinder.algorithm import backend as _backend
from cssfinder.io.asset_loader import State
from cssfinder.log import get_logger
from cssfinder.project.cssfproject import AlgoMode, Backend, Precision


class Gilbert:
    """Class interface of gilbert algorithm."""

    def __init__(
        self,
        initial: State,
        mode: AlgoMode,
        backend: Backend,
        precision: Precision,
        visibility: float,
    ) -> None:
        self.logger = get_logger()
        self.initial = initial
        self.mode = mode
        self.precision = precision
        self.visibility = visibility
        backend_type = _backend.new(backend, self.precision)
        self.backend = backend_type(self.initial, self.mode, self.visibility)

    def run(self, epochs: int, iterations: int, max_corrections: int) -> None:
        """Run epochs of iterations each, or up to max_corrections found."""
        total_iterations = epochs * iterations

        for epoch_index in range(epochs):
            self.logger.info(
                "Executing epoch {0} / {1} ({2:.2%})",
                epoch_index + 1,
                epochs,
                (epoch_index + 1) / epochs,
            )
            # Run N iterations of algorithm without checking stop conditions.
            self.backend.run_epoch(iterations, epoch_index)

            iterations_executed = (epoch_index + 1) * iterations
            self.logger.debug(
                "Executed {0} iterations, total {1} / {2} ({3:.2%})",
                iterations,
                iterations_executed,
                total_iterations,
                iterations_executed / total_iterations,
            )
            # Check if we already reached expected number of corrections
            if self.backend.corrections_count >= max_corrections:
                self.logger.info(
                    "Reached expected maximal number of corrections {}", max_corrections
                )
                break
