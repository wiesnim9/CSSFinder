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

import numpy as np
import numpy.typing as npt
from typing_extensions import Protocol


class Implementation(Protocol):
    """Implementation of Gilbert algorithm utilities with specific precision."""

    @staticmethod
    def product(
        matrix1: npt.NDArray[np.complex128], matrix2: npt.NDArray[np.complex128]
    ) -> np.float64:
        """Calculate scalar product of two matrices."""
        ...

    @staticmethod
    def normalize(mtx: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """Normalization of a vector."""
        ...

    @staticmethod
    def get_random_haar(depth: int) -> npt.NDArray[np.complex128]:
        """Generate a random vector with Haar measure."""
        ...

    @staticmethod
    def project(mtx1: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """Build a projection from a vector."""
        ...

    @staticmethod
    def random_d_fs(depth: int, quantity: int) -> npt.NDArray[np.complex128]:
        """Random n quDit state."""
        ...

    @staticmethod
    def optimize_d_fs(
        rho2: npt.NDArray[np.complex128],
        rho3: npt.NDArray[np.complex128],
        depth: int,
        quantity: int,
        epochs: int,
    ) -> npt.NDArray[np.complex128]:
        """Optimize implementation for FSnQd mode."""
        ...

    @staticmethod
    def random_unitary_d_fs(
        depth: int, quantity: int, idx: int
    ) -> npt.NDArray[np.complex128]:
        """N quDits."""
        ...

    @staticmethod
    def expand_d_fs(  # pylint: disable=invalid-name
        value: npt.NDArray[np.complex128],
        depth: int,
        quantity: int,
        idx: int,
    ) -> npt.NDArray[np.complex128]:
        """Expand an operator to n quDits."""
        ...

    @staticmethod
    def kronecker(
        mtx: npt.NDArray[np.complex128], mtx1: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        """Kronecker Product."""
        ...

    @staticmethod
    def rotate(
        rho2: npt.NDArray[np.complex128], unitary: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        """Sandwich an operator with a unitary."""
        ...
