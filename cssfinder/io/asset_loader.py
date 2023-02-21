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


"""This module contains implementation of asset loader class in version 1.0.0 which is
compatible with CSSFProject in version 1.0.0."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import reduce

import numpy as np
import numpy.typing as npt

from cssfinder.constants import PRIMES
from cssfinder.io.matrix import MatrixIO
from cssfinder.log import get_logger
from cssfinder.project.cssfproject import CSSFProject


class AssetLoader:
    """Provides interface for loading project assets from files."""

    def __init__(self, project: CSSFProject) -> None:
        self.project = project

    def load_initial_state(self, force_squash: bool = False) -> State:
        """Load initial state from file indicated by cssfproject configuration."""
        logger = get_logger()

        state_props = self.project.resources.initial_state
        # Replace magic variables with dynamically determined values
        state_matrix_file_path = self.project.expand_path(state_props.file)

        loader = MatrixIO.new(state_matrix_file_path)
        mtx = loader.load().astype(np.complex128)
        logger.info(
            "Loaded matrix from {0!r} of shape {1}", state_matrix_file_path, mtx.shape
        )

        # Forcefully reshape array to match shape criterion
        if force_squash:
            total_size = reduce(operator.mul, mtx.shape)
            one_axis_size = np.sqrt(total_size)
            mtx = mtx.reshape((one_axis_size, one_axis_size))

        # We are expecting loaded ndarray to be a square matrix, all other numbers of
        # dimensions cause crash.
        self._check_matrix_shape(mtx)

        total_size = len(mtx)

        depth, quantity = state_props.depth, state_props.quantity

        # Depth and quantity are optional and we can determine first matching pair
        # of those values based on number of rows in matrix
        if depth is None:
            old_quantity = quantity
            depth, quantity = self.detect_depth_and_quantity(total_size)

            if old_quantity is not None:
                logger.warning(
                    (
                        "State quantity ({0}) was ignored, as depth is None, "
                        + "quantity of {1} was deduced."
                    ),
                    old_quantity,
                    quantity,
                )

        elif quantity is None:
            quantity = self.detect_system_quantity(depth, total_size)

        logger.info(
            "Matrix represents system with depth = {0}, quantity = {1}", depth, quantity
        )
        return State(state=mtx, depth=depth, quantity=quantity)

    def _check_matrix_shape(self, mtx: npt.NDArray[np.complex128]) -> None:
        """Check if ndarray conforms shape rules."""
        logger = get_logger()

        if len(mtx.shape) == 2:
            pass

        elif len(mtx.shape) > 2:
            logger.critical(
                "Expected square matrix but got tensor with shape {0}", mtx.shape
            )
            raise NotExpectedTensor(mtx)

        elif len(mtx.shape) == 1:
            logger.critical(
                "Expected square matrix but got vector with shape {0}", mtx.shape
            )
            raise NotExpectedVector(mtx)

        elif len(mtx.shape) == 0:
            logger.critical("Expected square matrix but got scalar ({0})", mtx)
            raise NotExpectedScalar(mtx)

        else:
            raise AssertionError(mtx.shape)

        # Only square matrices are accepted.
        x_size, y_size = mtx.shape
        if x_size != y_size:
            logger.critical("Expected square matrix, but received shape {0}", mtx.shape)
            raise IncorrectMatrixShape(mtx)

    def detect_depth_and_quantity(self, total: int) -> tuple[int, int]:
        """Detect both system depth and system quantity.

        Parameters
        ----------
        total : int
            Total size of matrix.

        Returns
        -------
        tuple[int, int]
            Depth, quantity tuple.

        Raises
        ------
        ValueError
            When depth and quantity can't be determined.
        """
        for depth in PRIMES:
            quantity = int(math.log(total, depth))

            if quantity == int(quantity):
                get_logger().debug(
                    "Deduced quantity {0} and depth {1} when given total size {2}",
                    depth,
                    quantity,
                    total,
                )
                return depth, quantity

        raise ValueError(
            "Couldn't determine size of system, prime number range exceeded."
        )

    def detect_system_quantity(self, depth: int, total: int) -> int:
        """Detect system quantity (number of subsystems).

        Parameters
        ----------
        depth : int
            Depth of each subsystem (d)
        total : int
            Total size of matrix.

        Returns
        -------
        int
            Quantity.

        Raises
        ------
        ValueError
            When quantity can't be determined.
        """
        quantity = int(math.log(total, depth))

        if quantity == int(quantity):
            get_logger().debug(
                "Deduced quantity {0} when given depth of {1}", depth, quantity
            )
            return quantity

        raise ValueError(
            f"Couldn't determine size of system: {quantity} == {int(quantity)} is "
            "False.",
        )

    def load_symmetries(self) -> None:
        """Load matrices describing symmetries of system state."""

    def load_projection(self) -> None:
        """Load matrix describing projection of system state."""


@dataclass
class State:
    """State matrix and system dimensions."""

    state: npt.NDArray[np.complex128]
    """Path to file containing state matrix."""

    depth: int
    """Depth of system, ie.

    (D)imensions in qu(D)it. (d)
    """

    quantity: int
    """Quantity of systems.

    ie. number of qu(D)its in state. (n)
    """


class IncorrectMatrixShape(ValueError):
    """Raised when matrix has incorrect shape."""

    def __init__(self, mtx: npt.NDArray[np.complex128]) -> None:
        super().__init__()
        self.mtx = mtx


class NotExpectedTensor(IncorrectMatrixShape):
    """Raised when got 3+ dimensional tensor instead of matrix."""


class NotExpectedVector(IncorrectMatrixShape):
    """Raised when got vector instead of matrix."""


class NotExpectedScalar(IncorrectMatrixShape):
    """Raised when got scalar instead of matrix."""
