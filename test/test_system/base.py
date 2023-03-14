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

"""Module groups system level test of FSnQd mode of Gilbert algorithm."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from cssfinder.api import run_project_from
from cssfinder.io.gilbert_io import GilbertIO

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


class SetupRunProjectMixin:
    """Mixin class providing class setup running specific project."""

    PROJECT_PATH: Path
    TEST_TASK_NAME: str

    corrections: pd.DataFrame
    """List of corrections obtained from cssfinder."""

    state: npt.NDArray[np.complex128]
    """Final state matrix."""

    _temporary_directory: TemporaryDirectory

    @classmethod
    def get_project_directory(cls) -> Path:
        """Path to project directory."""
        return cls.get_temporary_directory() / cls.PROJECT_PATH.name

    @classmethod
    def get_output_directory(cls) -> Path:
        """Path to output directory."""
        return cls.get_project_directory() / "output" / cls.TEST_TASK_NAME

    @classmethod
    def get_temporary_directory(cls) -> Path:
        """Get path to temporary directory shared by tests in this class."""
        return Path(cls._temporary_directory.name)

    @classmethod
    def setup_class(cls) -> None:
        """Run class setup.

        Executed once for class, shared between tests within class.

        """
        cls._temporary_directory = TemporaryDirectory()
        shutil.copytree(
            cls.PROJECT_PATH.as_posix(), cls.get_project_directory().as_posix()
        )
        print(cls.get_project_directory())

        run_project_from(cls.get_project_directory(), [cls.TEST_TASK_NAME])

        gilbert_io = GilbertIO()

        cls.corrections = gilbert_io.load_corrections(
            cls.get_output_directory() / "corrections.json"
        )
        cls.state = gilbert_io.load_state(cls.get_output_directory() / "state.mtx")

    @classmethod
    def teardown_class(cls) -> None:
        """Clean up after class.

        Executed once for class, shared between tests within class.

        """


class ModeTest(SetupRunProjectMixin):
    """Base class for simple mode test suite."""

    @dataclass
    class MinMax:
        """Well named container for max and min values of floating range."""

        min: float  # noqa: A003  # not needed in this scope
        """Minimal value of range."""

        max: float  # noqa: A003  # not needed in this scope
        """Maximal value of range."""

    EXPECTED_MINIMAL_NUMBER_OF_CORRECTIONS: int

    OUT_STATE_ROW_COUNT: int
    OUT_STATE_COL_COUNT: int

    MIN_CORRECTION_VALUE: float
    MIN_MAX_FIRST_CORRECTION_RANGE: ModeTest.MinMax

    def test_number_of_corrections(self) -> None:
        """Check if valid number of corrections was saved."""
        assert len(self.corrections) >= self.EXPECTED_MINIMAL_NUMBER_OF_CORRECTIONS

    def test_first_correction(self) -> None:
        """Check if first correction value is within expected range."""
        value = self.corrections["value"].iloc[0]

        assert (
            self.MIN_MAX_FIRST_CORRECTION_RANGE.min
            < value
            < self.MIN_MAX_FIRST_CORRECTION_RANGE.max
        ), (
            self.MIN_MAX_FIRST_CORRECTION_RANGE.min,
            value,
            self.MIN_MAX_FIRST_CORRECTION_RANGE.max,
        )

    def test_last_better_than_first_correction(self) -> None:
        """Check if last correction is better (smaller) than first correction."""
        values = self.corrections["value"]
        first, last = values.iloc[0], values.iloc[-1]

        assert first > last > self.MIN_CORRECTION_VALUE

    def test_state_shape(self) -> None:
        """Check if output state has correct shape."""
        assert self.state.shape == (self.OUT_STATE_ROW_COUNT, self.OUT_STATE_COL_COUNT)

    def test_state_dtype(self) -> None:
        """Check if output state has correct data type."""
        assert self.state.dtype == np.complex128

    def test_state_not_all_real(self) -> None:
        """Check if there are complex parts of values in state."""
        assert not np.all(np.isreal(self.state))
