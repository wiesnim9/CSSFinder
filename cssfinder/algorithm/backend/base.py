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


"""Module contains base class for creating Gilbert algorithm backends (concrete
implementations) and exceptions which are expected to be raised from backends.
"""


from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from cssfinder.cssfproject import AlgoMode

BackendT = TypeVar("BackendT", bound="BackendBase")


class BackendBase:
    """Gilbert algorithm backend (implementation)."""

    author: ClassVar[str] = ""
    description: ClassVar[str] = ""

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
        self.depth = depth
        self.quantity = quantity
        self.initial = initial
        self.visibility = visibility
        self.mode = mode
        self.is_debug = is_debug

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
        raise NotImplementedError(self.set_symmetries.__qualname__)

    def set_projection(self, projection: npt.NDArray[np.complex128]) -> None:
        """Set projection to use during calculations.

        This operation may involve type conversion and copying of symmetries, therefore
        if may be slow and should should be done only once.

        Parameters
        ----------
        projection : npt.NDArray[np.complex128]
            Projection matrix.

        """
        raise NotImplementedError(self.set_projection.__qualname__)

    def get_state(self) -> npt.NDArray[np.complex128]:
        """Return current system state with all optimizations applied."""
        raise NotImplementedError(self.get_state.__qualname__)

    def get_corrections(self) -> list[tuple[int, int, float]]:
        """Return list of all corrections found during optimization."""
        raise NotImplementedError(self.get_corrections.__qualname__)

    def get_corrections_count(self) -> int:
        """Return number of all corrections found during optimization."""
        raise NotImplementedError(self.get_corrections_count.__qualname__)

    def run_epoch(self, iterations: int, epoch_index: int) -> None:
        """Run sequence of iterations without stopping to check any stop conditions."""
        raise NotImplementedError(self.run_epoch.__qualname__)
