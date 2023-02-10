"""This module contains."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cssfinder.algorithm.backend.base import BackendBase


class NumPy(BackendBase):
    """Implementation of Gilbert algorithm using python numpy library."""

    @property
    def state(self) -> npt.NDArray[np.complex128]:
        """Return current system state with all optimizations applied."""
        raise NotImplementedError()

    @property
    def corrections(self) -> list[tuple[int, int, float]]:
        """Return list of all corrections found during optimization."""
        raise NotImplementedError()

    @property
    def corrections_count(self) -> int:
        """Return number of all corrections found during optimization."""
        raise NotImplementedError()

    def run_epoch(self, iterations: int, epoch_index: int) -> None:
        """Run sequence of iterations without stopping to check any stop conditions."""
        raise NotImplementedError()
