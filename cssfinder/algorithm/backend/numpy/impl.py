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


"""Module contains declaration of Implementation protocol for numpy based backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

PRIMARY = TypeVar("PRIMARY", np.complex128, np.complex64)
SECONDARY_co = TypeVar("SECONDARY_co", np.float64, np.float32, covariant=True)


class Implementation(Generic[PRIMARY, SECONDARY_co], Protocol):
    """Implementation of Gilbert algorithm utilities with specific precision."""

    #  █████  ██████  ███    ███ ███    ███  ██████  ███    ██
    # ██     ██    ██ ████  ████ ████  ████ ██    ██ ████   ██
    # ██     ██    ██ ██ ████ ██ ██ ████ ██ ██    ██ ██ ██  ██
    # ██     ██    ██ ██  ██  ██ ██  ██  ██ ██    ██ ██  ██ ██
    #  █████  ██████  ██      ██ ██      ██  ██████  ██   ████

    @staticmethod
    def product(
        matrix1: npt.NDArray[PRIMARY],
        matrix2: npt.NDArray[PRIMARY],
    ) -> SECONDARY_co:
        """Calculate scalar product of two matrices."""
        ...

    @staticmethod
    def get_random_haar_1d(depth: int) -> npt.NDArray[PRIMARY]:
        """Generate a random vector with Haar measure."""
        ...

    @staticmethod
    def get_random_haar_2d(depth: int, quantity: int) -> npt.NDArray[PRIMARY]:
        """Generate multiple random vectors with Haar measure in form of matrix."""
        ...

    @staticmethod
    def normalize(mtx: npt.NDArray[PRIMARY]) -> npt.NDArray[PRIMARY]:
        """Normalize a vector."""
        ...

    @staticmethod
    def project(mtx1: npt.NDArray[PRIMARY]) -> npt.NDArray[PRIMARY]:
        """Build a projection from a vector."""
        ...

    @staticmethod
    def kronecker(
        mtx: npt.NDArray[PRIMARY],
        mtx1: npt.NDArray[PRIMARY],
    ) -> npt.NDArray[PRIMARY]:
        """Kronecker Product."""
        ...

    @staticmethod
    def rotate(
        rho2: npt.NDArray[PRIMARY],
        unitary: npt.NDArray[PRIMARY],
    ) -> npt.NDArray[PRIMARY]:
        """Sandwich an operator with a unitary."""
        ...

    #   ██████     ███████    ███████            ███    ███     ██████     ██████     ███████   # noqa: E501
    #   ██   ██    ██         ██                 ████  ████    ██    ██    ██   ██    ██        # noqa: E501
    #   ██   ██    █████      ███████            ██ ████ ██    ██    ██    ██   ██    █████     # noqa: E501
    #   ██   ██    ██              ██            ██  ██  ██    ██    ██    ██   ██    ██        # noqa: E501
    #   ██████     ██         ███████            ██      ██     ██████     ██████     ███████   # noqa: E501

    @staticmethod
    def optimize_d_fs(
        new_state: npt.NDArray[PRIMARY],
        visibility_state: npt.NDArray[PRIMARY],
        depth: int,
        quantity: int,
        epochs: int,
    ) -> npt.NDArray[PRIMARY]:
        """Optimize implementation for FSnQd mode."""
        ...

    @staticmethod
    def random_unitary_d_fs(
        depth: int,
        quantity: int,
        idx: int,
    ) -> npt.NDArray[PRIMARY]:
        """N quDits."""
        ...

    @staticmethod
    def random_d_fs(depth: int, quantity: int) -> npt.NDArray[PRIMARY]:
        """Draw random n quDit state."""
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

    #   ██████     ███████            ███    ███     ██████     ██████     ███████
    #   ██   ██    ██                 ████  ████    ██    ██    ██   ██    ██
    #   ██████     ███████            ██ ████ ██    ██    ██    ██   ██    █████
    #   ██   ██         ██            ██  ██  ██    ██    ██    ██   ██    ██
    #   ██████     ███████            ██      ██     ██████     ██████     ███████

    @staticmethod
    def random_bs(depth: int, quantity: int) -> npt.NDArray[PRIMARY]:
        """Draw random biseparable state."""
        ...

    @staticmethod
    def random_unitary_bs(depth: int, quantity: int) -> npt.NDArray[PRIMARY]:
        """Draw random unitary for biseparable state."""
        ...

    @staticmethod
    def random_unitary_bs_reverse(depth: int, quantity: int) -> npt.NDArray[PRIMARY]:
        """Draw random unitary for biseparable state."""
        ...

    @staticmethod
    def optimize_bs(
        new_state: npt.NDArray[PRIMARY],
        visibility_state: npt.NDArray[PRIMARY],
        depth: int,
        quantity: int,
        updates_count: int,
    ) -> npt.NDArray[PRIMARY]:
        """Run the minimization algorithm to optimize the biseparable state.

        Parameters
        ----------
        new_state : npt.NDArray[np.complex128]
            Randomly drawn state to be optimized.
        visibility_state : npt.NDArray[np.complex128]
            Visibility matrix.
        depth : int
            Depth of analyzed system.
        quantity : int
            Quantity of quDits in system.
        updates_count : int
            Number of optimizer iterations to execute.

        Returns
        -------
        npt.NDArray[np.complex128]
            Optimized state.

        """
        ...
