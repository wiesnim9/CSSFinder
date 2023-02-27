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


"""This file contains declaration of Implementation protocol for numpy based
backends."""

from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import Protocol

PRIMARY = TypeVar("PRIMARY", np.complex128, np.complex64)
SECONDARY_co = TypeVar("SECONDARY_co", np.float64, np.float32, covariant=True)


class Implementation(Generic[PRIMARY, SECONDARY_co], Protocol):
    """Implementation of Gilbert algorithm utilities with specific precision."""

    @staticmethod
    def product(
        matrix1: npt.NDArray[PRIMARY], matrix2: npt.NDArray[PRIMARY]
    ) -> SECONDARY_co:
        """Calculate scalar product of two matrices."""
        ...

    @staticmethod
    def normalize(mtx: npt.NDArray[PRIMARY]) -> npt.NDArray[PRIMARY]:
        """Normalization of a vector."""
        ...

    @staticmethod
    def get_random_haar(depth: int) -> npt.NDArray[PRIMARY]:
        """Generate a random vector with Haar measure."""
        ...

    @staticmethod
    def project(mtx1: npt.NDArray[PRIMARY]) -> npt.NDArray[PRIMARY]:
        """Build a projection from a vector."""
        ...

    @staticmethod
    def random_d_fs(depth: int, quantity: int) -> npt.NDArray[PRIMARY]:
        """Random n quDit state."""
        ...

    @staticmethod
    def optimize_d_fs(
        rho2: npt.NDArray[PRIMARY],
        rho3: npt.NDArray[PRIMARY],
        depth: int,
        quantity: int,
        epochs: int,
    ) -> npt.NDArray[PRIMARY]:
        """Optimize implementation for FSnQd mode."""
        ...

    @staticmethod
    def random_unitary_d_fs(
        depth: int, quantity: int, idx: int
    ) -> npt.NDArray[PRIMARY]:
        """N quDits."""
        ...

    @staticmethod
    def expand_d_fs(  # pylint: disable=invalid-name
        value: npt.NDArray[PRIMARY],
        depth: int,
        quantity: int,
        idx: int,
    ) -> npt.NDArray[PRIMARY]:
        """Expand an operator to n quDits."""
        ...

    @staticmethod
    def kronecker(
        mtx: npt.NDArray[PRIMARY], mtx1: npt.NDArray[PRIMARY]
    ) -> npt.NDArray[PRIMARY]:
        """Kronecker Product."""
        ...

    @staticmethod
    def rotate(
        rho2: npt.NDArray[PRIMARY], unitary: npt.NDArray[PRIMARY]
    ) -> npt.NDArray[PRIMARY]:
        """Sandwich an operator with a unitary."""
        ...
