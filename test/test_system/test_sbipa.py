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

from test.test_system.base import ModeTest

import pytest

from cssfinder.examples import Example


class Test_SBiPa(ModeTest):  # noqa: N801  # Underscore used for readability.
    """Test behavior of SBiPa mode of Gilbert algorithm."""

    EXPECTED_MINIMAL_NUMBER_OF_CORRECTIONS: int = 100
    PROJECT_PATH = Example.SBiPa_json.get_path()
    TEST_TASK_NAME: str = "test_sbipa_proj"

    OUT_STATE_ROW_COUNT: int = 9
    OUT_STATE_COL_COUNT: int = 9

    MIN_CORRECTION_VALUE: float = 0.001
    MIN_MAX_FIRST_CORRECTION_RANGE: ModeTest.MinMax = ModeTest.MinMax(0.001, 0.120)


class Test_SBiPa_WithProjection(  # noqa: N801  # Underscore used for readability.
    ModeTest
):
    """Test behavior of SBiPa mode of Gilbert algorithm with projections."""

    EXPECTED_MINIMAL_NUMBER_OF_CORRECTIONS: int = 100
    PROJECT_PATH = Example.SBiPa_json.get_path()
    TEST_TASK_NAME: str = "test_sbipa_proj_with_projection"

    OUT_STATE_ROW_COUNT: int = 9
    OUT_STATE_COL_COUNT: int = 9

    MIN_CORRECTION_VALUE: float = 0.001
    MIN_MAX_FIRST_CORRECTION_RANGE: ModeTest.MinMax = ModeTest.MinMax(0.001, 0.25)

    @pytest.mark.skip()
    def test_last_better_than_first_correction(self) -> None:
        """Check if last correction is better (smaller) than first correction."""
