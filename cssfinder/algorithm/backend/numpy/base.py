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


"""This module contains numpy based implementation of Gilbert algorithm but not bound to
specific precision."""

from __future__ import annotations
from typing import Generic, Type, TypeVar, cast

import numpy as np
import numpy.typing as npt

from numba import jit

from cssfinder.algorithm.backend.base import BackendBase
from cssfinder.algorithm.backend.numpy.impl import Implementation
from cssfinder.io.asset_loader import State
from cssfinder.project.cssfproject import AlgoMode


PRIMARY = TypeVar("PRIMARY", np.complex128, np.complex64, np.float64, np.float32)
SECONDARY_co = TypeVar("SECONDARY_co", np.float64, np.float32, covariant=True)


class NumPyBase(Generic[PRIMARY, SECONDARY_co], BackendBase):
    """Implementation of Gilbert algorithm using python numpy library."""

    _visibility: npt.NDArray[PRIMARY]
    _intermediate: npt.NDArray[PRIMARY]
    _visibility_reduced: npt.NDArray[PRIMARY]

    _corrections: list[tuple[int, int, float]]

    impl: Implementation[PRIMARY, SECONDARY_co]
    primary_t: Type[PRIMARY]
    secondary_t: Type[SECONDARY_co]

    def __init__(self, initial: State, mode: AlgoMode, visibility: float) -> None:
        super().__init__(initial, mode, visibility)

        self._visibility = self._create_visibility_matrix()
        self._intermediate = self._create_intermediate_state()
        self._corrections = []

        self._aa4: SECONDARY_co = 2 * self.impl.product(
            self._visibility, self._intermediate
        )
        self._aa6: SECONDARY_co = self.impl.product(
            self._intermediate, self._intermediate
        )
        self._visibility_reduced = cast(
            npt.NDArray[PRIMARY], (self._visibility - self._intermediate)
        )
        self._dd1: SECONDARY_co = self.impl.product(
            self._intermediate, self._visibility_reduced
        )

    def _create_visibility_matrix(self) -> npt.NDArray[PRIMARY]:
        vis_state = self.visibility * self.initial.state
        inv_vis_ident = (1 - self.visibility) * np.identity(
            len(self.initial.state), dtype=np.complex128
        )
        return (vis_state + inv_vis_ident / len(self.initial.state)).astype(
            self.primary_t
        )

    def _create_intermediate_state(self) -> npt.NDArray[PRIMARY]:
        intermediate = np.zeros_like(self._visibility, dtype=np.complex128)
        np.fill_diagonal(intermediate, self._visibility.diagonal())
        return intermediate.astype(self.primary_t)

    @property
    def state(self) -> npt.NDArray[np.complex128]:
        """Return current system state with all optimizations applied."""
        return self._intermediate.copy().astype(np.complex128)

    @property
    def corrections(self) -> list[tuple[int, int, float]]:
        """Return list of all corrections found during optimization."""
        return self._corrections.copy()

    @property
    def corrections_count(self) -> int:
        """Return number of all corrections found during optimization."""
        return len(self._corrections)

    @jit(forceobj=True, cache=True, looplift=False)
    def run_epoch(self, iterations: int, epoch_index: int) -> None:
        """Run sequence of iterations without stopping to check any stop conditions."""

        depth = self.initial.depth
        quantity = self.initial.quantity
        epochs = 20 * depth * depth * quantity

        for iteration_index in range(iterations):
            alternative_state = self.impl.random_d_fs(depth, quantity)

            if (
                self.impl.product(alternative_state, self._visibility_reduced)
                > self._dd1
            ):
                self._update_state(
                    alternative_state, iterations, epoch_index, epochs, iteration_index
                )

    @jit(forceobj=True, cache=True, looplift=False, inline="always")
    def _update_state(
        self,
        alternative_state: npt.NDArray[PRIMARY],
        iterations: int,
        epoch_index: int,
        epochs: int,
        iteration_index: int,
    ) -> None:
        depth = self.initial.depth
        quantity = self.initial.quantity

        alternative_state = self.impl.optimize_d_fs(
            alternative_state, self._visibility_reduced, depth, quantity, epochs
        )

        aa3: SECONDARY_co = self.impl.product(alternative_state, alternative_state)
        aa2: SECONDARY_co = 2 * self.impl.product(self._visibility, alternative_state)
        aa5: SECONDARY_co = 2 * self.impl.product(self._intermediate, alternative_state)

        param: SECONDARY_co = -(-self._aa4 + aa2 + aa5 - 2 * aa3) / (
            2 * (self._aa6 - aa5 + aa3)
        )

        if 0 <= param <= 1:
            inv_alternative_state: npt.NDArray[PRIMARY] = cast(
                npt.NDArray[PRIMARY], ((1 - param) * alternative_state)
            )
            self._intermediate = cast(
                npt.NDArray[PRIMARY],
                (param * self._intermediate) + inv_alternative_state,
            )

            self._visibility_reduced = cast(
                npt.NDArray[PRIMARY], (self._visibility - self._intermediate)
            )
            self._aa4 = 2 * self.impl.product(self._visibility, self._intermediate)
            self._aa6 = self.impl.product(self._intermediate, self._intermediate)
            self._dd1 = self._aa4 / 2 - self._aa6

            self._corrections.append(
                (
                    epoch_index * iterations + iteration_index,
                    len(self._corrections),
                    float(
                        self.impl.product(
                            self._visibility_reduced,
                            self._visibility_reduced,
                        )
                    ),
                )
            )
