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


"""This module contains base class for creating Gilbert algorithm backends (concrete
implementations) and exceptions which are expected to be raised from backends."""


from __future__ import annotations

from typing import Type, TypeVar

import numpy as np
import numpy.typing as npt

from cssfinder.cssfproject import AlgoMode, Backend, Precision
from cssfinder.io.asset_loader import State

BackendT = TypeVar("BackendT", bound="BackendBase")


class BackendBase:
    """Gilbert algorithm backend (implementation)."""

    backend_index: dict[tuple[Backend, Precision], Type[BackendBase]] = {}

    def __init__(self, initial: State, mode: AlgoMode, visibility: float) -> None:
        self.initial = initial
        self.visibility = visibility
        self.mode = mode

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
