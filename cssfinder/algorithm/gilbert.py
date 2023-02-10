"""This module contains public interface of Gilbert algorithm."""

from __future__ import annotations

from cssfinder.algorithm.backend.base import BackendBase
from cssfinder.io.v1_0_0.asset_loader import State
from cssfinder.log import get_logger
from cssfinder.project.v1_0_0.cssfproject import AlgoMode, Backend, Precision


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
        backend_type = BackendBase.select(backend, self.precision)
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
                self.logger.info("")
                break
