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


"""Program input/output utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import IO, TYPE_CHECKING, ClassVar

import numpy as np
import scipy.io

import cssfinder

if TYPE_CHECKING:
    import numpy.typing as npt


class MatrixFormat(Enum):
    """Enumeration of possible supported matrix formats."""

    MATRIX_MARKET = ".mtx"


class MatrixIO(ABC):
    """Abstract base class declaring interface of matrix input-output manager."""

    matrix_format: ClassVar[MatrixFormat]

    @classmethod
    def new(cls, matrix_format: MatrixFormat) -> MatrixIO:
        """Create new instance of matrix loader for file format. When format is None,
        file format is deduced from file extension.

        Parameters
        ----------
        matrix_format : MatrixFormat
            Path to file containing matrix data.

        Returns
        -------
        MatrixIO
            Matrix loader.

        """
        io = FORMAT_TO_LOADER[matrix_format]()
        logging.debug("Selected matrix IO %r for format %r", io, matrix_format.name)
        return io

    def load(
        self, src: str | Path | IO[bytes]
    ) -> npt.NDArray[np.int64 | np.float64 | np.complex128]:
        """Load matrix from file as numpy array."""
        if isinstance(src, (str, Path)):
            with Path(src).open("rb") as file:
                return self._load(file)

        return self._load(file)

    @abstractmethod
    def _load(
        self, src: IO[bytes]
    ) -> npt.NDArray[np.int64 | np.float64 | np.complex128]:
        ...

    def dump(
        self,
        data: npt.NDArray[np.int64 | np.float64 | np.complex128],
        dest: str | Path | IO[bytes],
    ) -> None:
        """Dump matrix to file from numpy array."""
        if isinstance(dest, (str, Path)):
            file_path = Path(dest).with_suffix(self.matrix_format.value)
            with file_path.open("wb") as file:
                return self._dump(data, file)

        return self._dump(data, file)

    @abstractmethod
    def _dump(
        self, data: npt.NDArray[np.int64 | np.float64 | np.complex128], dest: IO[bytes]
    ) -> None:
        ...


class MatrixMarketIO(MatrixIO):
    """MatrixIO implementation for loading MatrixMarket exchange format files."""

    matrix_format: ClassVar[MatrixFormat] = MatrixFormat.MATRIX_MARKET

    def _load(
        self, dest: IO[bytes]
    ) -> npt.NDArray[np.int64 | np.float64 | np.complex128]:
        mtx = scipy.io.mmread(dest)
        assert mtx is not None
        return np.array(mtx)

    def _dump(
        self, data: npt.NDArray[np.int64 | np.float64 | np.complex128], dest: IO[bytes]
    ) -> None:
        scipy.io.mmwrite(
            dest,
            data,
            comment=f"Created with CSSFinder {cssfinder.__version__}.",
        )


FORMAT_TO_LOADER: dict[MatrixFormat, type[MatrixIO]] = {
    MatrixFormat.MATRIX_MARKET: MatrixMarketIO,
}
