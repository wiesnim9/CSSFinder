"""Program input/output utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type

import numpy as np
import scipy.io

from cssfinder.log import get_logger
from cssfinder.types import MatrixT


class MatrixIO(ABC):
    """Abstract base class declaring interface of matrix input-output manager."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    @classmethod
    def new(cls, file_path: str | Path, file_format: Optional[str] = None) -> MatrixIO:
        """Create new instance of matrix loader for file format. When format is None,
        file format is deduced from file extension.

        Parameters
        ----------
        file_path : Path
            Path to file containing matrix data.
        format : Optional[str], optional
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

        logger = get_logger()
        logger.debug(
            "Using file format {!r} for file {}", file_format, file_path.as_posix()
        )

        return FORMAT_TO_LOADER[file_format](file_path)

    @abstractmethod
    def load(self) -> MatrixT:
        """Load matrix from file as numpy array."""


class MatrixMarketIO(MatrixIO):
    """MatrixIO implementation for loading MatrixMarket exchange format files."""

    def load(self) -> MatrixT:
        mtx = scipy.io.mmread(self.file_path.as_posix())
        assert mtx is not None
        return np.array(mtx)


FORMAT_TO_LOADER: dict[str, Type[MatrixIO]] = {
    ".mtx": MatrixMarketIO,
}
