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

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pandas as pd

from cssfinder.io.matrix import MatrixFormat, MatrixIO

if TYPE_CHECKING:
    import numpy.typing as npt

NUMBER_OF_DIMENSIONS_IN_MATRIX: int = 2


class GilbertIO:
    """Loader of Gilbert algorithm assets."""

    def __init__(
        self, matrix_format: MatrixFormat = MatrixFormat.MATRIX_MARKET
    ) -> None:
        self.loader = MatrixIO.new(matrix_format)

    def load_state(self, source: str | Path) -> npt.NDArray[np.complex128]:
        """Load state matrix from file.

        Parameters
        ----------
        source : str | Path
            Path to matrix file.

        Returns
        -------
        npt.NDArray[np.complex128]
            Loaded matrix. Always returns np.complex128.

        """
        mtx = self.loader.load(source).astype(np.complex128)
        logging.info(
            "Loaded matrix from %r of shape %r", Path(source).as_posix(), mtx.shape
        )

        # We are expecting loaded ndarray to be a square matrix, all other numbers of
        # dimensions cause crash.
        self._check_matrix_shape(mtx)

        return mtx

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

    def load_symmetries(
        self,
        symmetries: Optional[list[list[str]]],
    ) -> list[list[npt.NDArray[np.complex128]]]:
        """Load matrices describing symmetries of system state."""
        if symmetries is None:
            return []
        return [
            [self.loader.load(sym).astype(np.complex128) for sym in row]
            for row in symmetries
        ]

    def load_projection(
        self,
        projection: Optional[str],
    ) -> npt.NDArray[np.complex128] | None:
        """Load matrix describing projection of system state."""
        if projection is None:
            return None

        return self.loader.load(projection).astype(np.complex128)

    def dump_state(self, state: npt.NDArray[np.complex128], dest: str | Path) -> None:
        """Save state to file."""
        return self.loader.dump(state, dest)

    def dump_corrections(self, corrections: Any, dest: str | Path) -> None:
        """Save state to file."""
        file_path = Path(dest).with_suffix(".json")
        with file_path.open("w", encoding="utf-8") as file:
            return json.dump(corrections, file)

    def load_corrections(self, source: Path) -> pd.DataFrame:
        """Load corrections from a JSON file and return them as a pandas DataFrame.

        Parameters
        ----------
        source : Path
            Path to the JSON file containing corrections data.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the corrections data, with columns renamed to
            "iteration", "index", and "value".

        """
        data_frame: pd.DataFrame = cast(pd.DataFrame, pd.read_json(source))

        return data_frame.rename(
            columns={0: "iteration", 1: "index", 2: "value"},
        )


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
