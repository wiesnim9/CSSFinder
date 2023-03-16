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


"""Module contains numpy based implementation of Gilbert algorithm but not bound to
specific precision.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, cast

import numpy as np
import numpy.typing as npt
from numba import jit

from cssfinder.algorithm.backend.base import BackendBase
from cssfinder.cssfproject import AlgoMode

if TYPE_CHECKING:
    from cssfinder.algorithm.backend.numpy.impl import Implementation

PRIMARY = TypeVar("PRIMARY", np.complex128, np.complex64)
SECONDARY_co = TypeVar("SECONDARY_co", np.float64, np.float32, covariant=True)


class NumPyBase(Generic[PRIMARY, SECONDARY_co], BackendBase):
    """Implementation of Gilbert algorithm using python numpy library."""

    _visibility: npt.NDArray[PRIMARY]
    _intermediate: npt.NDArray[PRIMARY]
    _visibility_reduced: npt.NDArray[PRIMARY]

    _corrections: list[tuple[int, int, float]]
    _symmetries: list[list[npt.NDArray[PRIMARY]]]
    _projection: Optional[npt.NDArray[PRIMARY]]

    impl: Implementation[PRIMARY, SECONDARY_co]
    primary_t: type[PRIMARY]
    secondary_t: type[SECONDARY_co]

    def __init__(  # noqa: PLR0913
        self,
        initial: npt.NDArray[np.complex128],
        depth: int,
        quantity: int,
        mode: AlgoMode,
        visibility: float,
        *,
        is_debug: bool = False,
    ) -> None:
        super().__init__(initial, depth, quantity, mode, visibility, is_debug=is_debug)

        self.initial = initial.astype(self.primary_t)

        self._visibility = self._create_visibility_matrix()
        self._intermediate = self._create_intermediate_state()
        self._corrections = []

        self._symmetries = []
        self._projection = None

        self._aa4: SECONDARY_co = 2 * self.impl.product(
            self._visibility,
            self._intermediate,
        )
        self._aa6: SECONDARY_co = self.impl.product(
            self._intermediate,
            self._intermediate,
        )
        self._visibility_reduced = cast(
            npt.NDArray[PRIMARY],
            (self._visibility - self._intermediate),
        )
        self._dd1: SECONDARY_co = self.impl.product(
            self._intermediate,
            self._visibility_reduced,
        )

        if not self.is_debug:
            self.jit()

    def _create_visibility_matrix(self) -> npt.NDArray[PRIMARY]:
        vis_state = self.visibility * self.initial
        inv_vis_ident = (1 - self.visibility) * np.identity(
            len(self.initial),
            dtype=np.complex128,
        )
        return (vis_state + inv_vis_ident / len(self.initial)).astype(
            self.primary_t,
        )

    def _create_intermediate_state(self) -> npt.NDArray[PRIMARY]:
        intermediate = np.zeros_like(self._visibility, dtype=np.complex128)
        np.fill_diagonal(intermediate, self._visibility.diagonal())
        return intermediate.astype(self.primary_t)

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
        self._symmetries = [
            [cell.astype(self.primary_t) for cell in row] for row in symmetries
        ]
        if self._symmetries:
            self._intermediate = self.impl.apply_symmetries(
                self._intermediate, self._symmetries
            )

    def set_projection(self, projection: npt.NDArray[np.complex128]) -> None:
        """Set projection to use during calculations.

        This operation may involve type conversion and copying of symmetries, therefore
        if may be slow and should should be done only once.

        Parameters
        ----------
        projection : npt.NDArray[np.complex128]
            Projection matrix.

        """
        self._projection = projection.astype(self.primary_t)
        self._intermediate = self.impl.rotate(self._intermediate, self._projection)
        logging.debug("Projection have been set %r", self._projection)

    def get_state(self) -> npt.NDArray[np.complex128]:
        """Return current system state with all optimizations applied."""
        return self._intermediate.copy().astype(np.complex128)

    def get_corrections(self) -> list[tuple[int, int, float]]:
        """Return list of all corrections found during optimization."""
        return self._corrections.copy()

    def get_corrections_count(self) -> int:
        """Return number of all corrections found during optimization."""
        return len(self._corrections)

    def jit(self) -> None:
        """JIT compile performance critical parts of backend with numba."""
        _update_state = jit(  # type: ignore[assignment]
            forceobj=True,
            cache=True,
            looplift=False,
        )(
            self.__class__._update_state,  # noqa: SLF001
        )

        self._update_state = MethodType(_update_state, self)  # type: ignore[assignment]

        run_epoch = jit(forceobj=True, cache=True, looplift=False)(
            self.__class__.run_epoch,
        )

        self.run_epoch = MethodType(run_epoch, self)  # type: ignore[assignment]

    def run_epoch(self, iterations: int, epoch_index: int) -> None:
        """Run sequence of iterations without stopping to check any stop conditions."""
        depth = self.depth
        quantity = self.quantity
        epochs = 20 * depth * depth * quantity

        for iteration_index in range(iterations):
            if self.mode == AlgoMode.FSnQd:
                alternative_state = self.impl.random_d_fs(depth, quantity)
            elif self.mode == AlgoMode.SBiPa:
                alternative_state = self.impl.random_bs(depth, quantity)

            if (
                self.impl.product(alternative_state, self._visibility_reduced)
                > self._dd1
            ):
                self._update_state(
                    alternative_state,
                    iterations,
                    epoch_index,
                    epochs,
                    iteration_index,
                )

    def _update_state(  # noqa: PLR0913
        self,
        alternative_state: npt.NDArray[PRIMARY],
        iterations: int,
        epoch_index: int,
        epochs: int,
        iteration_index: int,
    ) -> None:
        depth = self.depth
        quantity = self.quantity

        if self.mode == AlgoMode.FSnQd:
            alternative_state = self.impl.optimize_d_fs(
                alternative_state,
                self._visibility_reduced,
                depth,
                quantity,
                epochs,
            )
        elif self.mode == AlgoMode.SBiPa:
            alternative_state = self.impl.optimize_bs(
                alternative_state,
                self._visibility_reduced,
                depth,
                quantity,
                epochs,
            )

        if self._symmetries:
            self._intermediate = self.impl.apply_symmetries(
                self._intermediate, self._symmetries
            )
        if self._projection is not None:
            self._intermediate = self.impl.rotate(self._intermediate, self._projection)

        aa3: SECONDARY_co = self.impl.product(alternative_state, alternative_state)
        aa2: SECONDARY_co = 2 * self.impl.product(self._visibility, alternative_state)
        aa5: SECONDARY_co = 2 * self.impl.product(self._intermediate, alternative_state)

        bb2: SECONDARY_co = -self._aa4 + aa2 + aa5 - 2 * aa3
        bb3: SECONDARY_co = self._aa6 - aa5 + aa3
        cc1: SECONDARY_co = -bb2 / (2 * bb3)

        if 0 <= cc1 <= 1:
            self._intermediate = cast(
                npt.NDArray[PRIMARY],
                (cc1 * self._intermediate) + ((1 - cc1) * alternative_state),
            )

            self._visibility_reduced = cast(
                npt.NDArray[PRIMARY],
                (self._visibility - self._intermediate),
            )
            self._aa4 = 2 * self.impl.product(self._visibility, self._intermediate)
            self._aa6 = self.impl.product(self._intermediate, self._intermediate)
            self._dd1 = self._aa4 / 2 - self._aa6

            self._corrections.append(
                (
                    epoch_index * iterations + iteration_index + 1,
                    len(self._corrections) + 1,
                    float(
                        self.impl.product(
                            self._visibility_reduced,
                            self._visibility_reduced,
                        ),
                    ),
                ),
            )
