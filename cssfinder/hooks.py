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

"""This module contains basic hooks which can be used by tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from cssfinder.io.matrix import MatrixIO


def save_matrix_hook(
    destination: str | Path,
) -> Callable[[npt.NDArray[np.complex128]], None]:
    """Builtin hook for saving state matrix.

    Matrix will be saved every time hook is called.
    """
    destination_path = Path(destination)
    mtx_io = MatrixIO.new(destination_path)

    def _save_matrix_hook(matrix: npt.NDArray[np.complex128]) -> None:
        logging.debug(
            "Saving output state to %r with matrix size %r",
            destination_path.as_posix(),
            matrix.shape,
        )
        mtx_io.dump(matrix)

    return _save_matrix_hook


def save_corrections_hook(
    destination: str | Path,
) -> Callable[[list[tuple[int, int, float]]], None]:
    """Builtin hook for saving corrections array.

    Corrections will be saved every time hook is called.
    """
    destination_path = Path(destination)

    def _save_corrections_hook(corrections: list[tuple[int, int, float]]) -> None:
        logging.debug(
            "Saving output state to %r with total corrections %r",
            destination_path.as_posix(),
            len(corrections),
        )
        with destination_path.open("w", encoding="utf-8") as file:
            json.dump(corrections, file, indent=2)

    return _save_corrections_hook
