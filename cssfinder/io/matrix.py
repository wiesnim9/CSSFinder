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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io

import cssfinder

if TYPE_CHECKING:
    import numpy.typing as npt


class MatrixIO(ABC):
    """Abstract base class declaring interface of matrix input-output manager."""

    def __init__(self, file_path: Path) -> None:
        """Initialize matrix IO.

        Parameters
        ----------
        file_path : Path
            Path to source/destination file.

        """
        self.file_path = file_path

    @classmethod
    def new(cls, file_path: str | Path, file_format: str | None = None) -> MatrixIO:
        """Create new instance of matrix loader for file format. When format is None,
        file format is deduced from file extension.

        Parameters
        ----------
        file_path : Path
            Path to file containing matrix data.
        file_format : Optional[str], optional
            File format specifier in form of extension, eg. ".mtx", by default None

        Returns
        -------
        MatrixIO
            Matrix loader.

        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if file_format is None:
            file_format = file_path.suffix

        logging.debug(
            "Using file format %r for file %r",
            file_format,
            file_path.as_posix(),
        )

        return FORMAT_TO_LOADER[file_format](file_path)

    @abstractmethod
    def load(self) -> npt.NDArray[np.int64 | np.float64 | np.complex128]:
        """Load matrix from file as numpy array."""

    @abstractmethod
    def dump(self, data: npt.NDArray[np.int64 | np.float64 | np.complex128]) -> None:
        """Dump matrix to file from numpy array."""


class MatrixMarketIO(MatrixIO):
    """MatrixIO implementation for loading MatrixMarket exchange format files."""

    def load(self) -> npt.NDArray[np.int64 | np.float64 | np.complex128]:
        """Load the data from MatrixMarket exchange format file.

        Returns
        -------
        numpy.ndarray
            An array containing the data from the file in MatrixMarket format.

        """
        mtx = scipy.io.mmread(self.file_path.as_posix())
        assert mtx is not None
        return np.array(mtx)

    def dump(self, data: npt.NDArray[np.int64 | np.float64 | np.complex128]) -> None:
        """Write data to a MatrixMarket exchange format file.

        Parameters
        ----------
        data : numpy.ndarray
            An array containing the data to be written to the file.

        Returns
        -------
        None

        """
        self.file_path.touch(0o664, exist_ok=True)
        scipy.io.mmwrite(
            self.file_path.as_posix(),
            data,
            comment=f"Created with CSSFinder {cssfinder.__version__}.",
        )


FORMAT_TO_LOADER: dict[str, type[MatrixIO]] = {
    ".mtx": MatrixMarketIO,
}
