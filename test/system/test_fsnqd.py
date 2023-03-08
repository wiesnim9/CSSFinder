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

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cssfinder.api import run_project_from
from cssfinder.io.output_loader import GilbertOutputLoader

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

PROJECT_5QUBITS = Path.cwd() / "examples" / "5qubits"


class TestFSnQd5qubits:
    """Test behavior of FSnQd mode of Gilbert algorithm."""

    EXPECTED_MINIMAL_NUMBER_OF_CORRECTIONS: int = 10

    output: Path
    """Path to output directory."""

    corrections: pd.DataFrame
    """List of corrections obtained from cssfinder."""

    state: npt.NDArray[np.complex128]
    """Final state matrix."""

    @classmethod
    def setup_class(cls) -> None:
        """Run class setup.

        Executed once for class, shared between tests within class.

        """
        run_project_from(PROJECT_5QUBITS, ["test_fsnqd_5qubits"])
        cls.output = PROJECT_5QUBITS / "output" / "test_fsnqd_5qubits"

        cls.corrections = GilbertOutputLoader().load_corrections_from(
            cls.output / "corrections.json"
        )
        cls.state = GilbertOutputLoader().load_state_from(cls.output / "state.mtx")

    def test_number_of_corrections(self) -> None:
        """Check if valid number of corrections was saved."""
        assert len(self.corrections) >= self.EXPECTED_MINIMAL_NUMBER_OF_CORRECTIONS

    def test_first_correction(self) -> None:
        """Check if first correction value is within expected range."""
        value = self.corrections["value"].iloc[0]

        min_value = 0.118
        max_value = 0.120

        assert min_value < value < max_value

    def test_last_better_than_first_correction(self) -> None:
        """Check if last correction is better (smaller) than first correction."""
        values = self.corrections["value"]
        first, last = values.iloc[0], values.iloc[-1]

        min_expected = 0.100

        assert first > last > min_expected

    def test_state_shape(self) -> None:
        """Check if output state has correct shape."""
        state_row_count = 32
        state_column_count = 32

        assert self.state.shape == (state_row_count, state_column_count)

    def test_state_dtype(self) -> None:
        """Check if output state has correct data type."""
        assert self.state.dtype == np.complex128

    def test_state_not_all_real(self) -> None:
        """Check if there are complex parts of values in state."""
        assert not np.all(np.isreal(self.state))
