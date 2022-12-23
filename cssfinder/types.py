from __future__ import annotations
from typing import TYPE_CHECKING

import numpy.typing as npt
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

DTypeT: TypeAlias = "np.int64 | np.float64 | np.complex128"

MtxT: TypeAlias = npt.NDArray[DTypeT]

MtxC128T: TypeAlias = npt.NDArray[np.complex128]
