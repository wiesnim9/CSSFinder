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


"""Module contains implementation of asset loader class."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt  # noqa: TCH002

from cssfinder.base_model import CommonBaseModel
from cssfinder.constants import PRIMES
from cssfinder.io.matrix import MatrixIO

if TYPE_CHECKING:
    from cssfinder.cssfproject import GilbertCfg

NUMBER_OF_DIMENSIONS_IN_MATRIX: int = 2


class GilbertAssets(CommonBaseModel):
    """Container class for assets used by gilbert algorithm."""

    state: State
    """Initial state for algorithm."""

    symmetries: Optional[list[npt.NDArray[np.complex128]]]
    """List of symmetries of state."""

    projection: Optional[npt.NDArray[np.complex128]]
    """Projection to apply to state."""


class GilbertAssetLoader:
    """Loader of Gilbert algorithm assets."""

    def load_assets(self, gilbert_cfg: GilbertCfg) -> GilbertAssets:
        """Load initial state from file indicated by cssfproject configuration."""
        return GilbertAssets(
            state=self._load_state(gilbert_cfg),
            symmetries=self._load_symmetries(gilbert_cfg),
            projection=self._load_projection(gilbert_cfg),
        )

    def _load_state(self, gilbert_cfg: GilbertCfg) -> State:
        state_props = gilbert_cfg.get_state()
        state_matrix_file_path = state_props.file

        loader = MatrixIO.new(state_matrix_file_path)
        mtx = loader.load().astype(np.complex128)
        logging.info(
            "Loaded matrix from %r of shape %r",
            state_matrix_file_path,
            mtx.shape,
        )

        # We are expecting loaded ndarray to be a square matrix, all other numbers of
        # dimensions cause crash.
        self._check_matrix_shape(mtx)

        total_size = len(mtx)
        depth, quantity = state_props.depth, state_props.quantity

        # Depth and quantity are optional and we can determine first matching pair
        # of those values based on number of rows in matrix
        if depth is None:
            old_quantity = quantity
            depth, quantity = self._detect_depth_and_quantity(total_size)

            if old_quantity is not None:
                logging.warning(
                    "State quantity (%r) was ignored, as depth is None, quantity of %r \
                        was deduced.",
                    old_quantity,
                    quantity,
                )

        elif quantity is None:
            quantity = self._detect_system_quantity(depth, total_size)

        logging.info(
            "Matrix represents system with depth = %r, quantity = %r",
            depth,
            quantity,
        )
        return State(state=mtx, depth=depth, quantity=quantity)

    def _check_matrix_shape(self, mtx: npt.NDArray[np.complex128]) -> None:
        """Check if ndarray conforms shape rules."""
        if len(mtx.shape) == NUMBER_OF_DIMENSIONS_IN_MATRIX:
            pass

        elif len(mtx.shape) > NUMBER_OF_DIMENSIONS_IN_MATRIX:
            logging.critical(
                "Expected square matrix but got tensor with shape %r",
                mtx.shape,
            )
            raise NotExpectedTensorError(mtx)

        elif len(mtx.shape) == 1:
            logging.critical(
                "Expected square matrix but got vector with shape %r",
                mtx.shape,
            )
            raise NotExpectedVectorError(mtx)

        elif len(mtx.shape) == 0:
            logging.critical("Expected square matrix but got scalar (%r)", mtx)
            raise NotExpectedScalarError(mtx)

        else:
            raise AssertionError(mtx.shape)

        # Only square matrices are accepted.
        x_size, y_size = mtx.shape
        if x_size != y_size:
            logging.critical("Expected square matrix, but received shape %r", mtx.shape)
            raise IncorrectMatrixShapeError(mtx)

    def _detect_depth_and_quantity(self, total: int) -> tuple[int, int]:
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
                logging.debug(
                    "Deduced quantity %r and depth %r when given total size %r",
                    depth,
                    quantity,
                    total,
                )
                return depth, quantity

        reason = "prime number range exceeded"
        raise UndefinedSystemSizeError(reason)

    def _detect_system_quantity(self, depth: int, total: int) -> int:
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
            logging.debug("Deduced quantity %r when given depth of %r", depth, quantity)
            return quantity

        error_reason = f"{quantity} != {int(quantity)}"
        raise UndefinedSystemSizeError(error_reason)

    def _load_symmetries(
        self,
        gilbert_cfg: GilbertCfg,  # noqa: ARG002
    ) -> list[npt.NDArray[np.complex128]] | None:
        """Load matrices describing symmetries of system state."""
        return  # type: ignore[return-value]

    def _load_projection(
        self,
        gilbert_cfg: GilbertCfg,  # noqa: ARG002
    ) -> npt.NDArray[np.complex128] | None:
        """Load matrix describing projection of system state."""
        return  # type: ignore[return-value]


@dataclass
class State:
    """State matrix and system dimensions."""

    state: npt.NDArray[np.complex128]
    """Path to file containing state matrix."""

    depth: int
    """Depth of system, ie.

    Dimensions in qu(D)it. (d)

    """

    quantity: int
    """Quantity of systems.

    ie. number of qu(D)its in state. (n)

    """


class IncorrectMatrixShapeError(ValueError):
    """Raised when matrix has incorrect shape."""

    def __init__(self, mtx: npt.NDArray[np.complex128]) -> None:
        """Store matrix object in `mtx` attribute."""
        super().__init__()
        self.mtx = mtx


class NotExpectedTensorError(IncorrectMatrixShapeError):
    """Raised when got 3+ dimensional tensor instead of matrix."""


class NotExpectedVectorError(IncorrectMatrixShapeError):
    """Raised when got vector instead of matrix."""


class NotExpectedScalarError(IncorrectMatrixShapeError):
    """Raised when got scalar instead of matrix."""


class UndefinedSystemSizeError(ValueError):
    """Raised when it is not possible to determine system dimensions."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Couldn't determine size of system: {reason}.")


GilbertAssets.update_forward_refs()
