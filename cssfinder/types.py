"""This module contains helper types for CSSFinder package (EOL module)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

DTypeT: TypeAlias = "np.int64 | np.float64 | np.complex128"

MatrixT: TypeAlias = npt.NDArray[DTypeT]

MatrixC128T: TypeAlias = npt.NDArray[np.complex128]
