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

"""Tests validating reports generation."""

from __future__ import annotations

from pathlib import Path
from test.test_system.base import SetupRunProjectMixin
from typing import TYPE_CHECKING

from cssfinder.api import create_report_from

if TYPE_CHECKING:
    from cssfinder.reports.renderer import ReportType


class ReportTestBase(SetupRunProjectMixin):
    """Validate report behavior."""

    PROJECT_PATH = Path.cwd() / "examples" / "5qubits"
    TEST_TASK_NAME: str = "test_fsnqd_5qubits"
    REPORT_TYPE: ReportType

    def generate_report(self, report_type: ReportType) -> None:
        """Generate report."""
        for report in create_report_from(
            self.get_project_directory(), self.TEST_TASK_NAME, [report_type]
        ):
            report.save_default()

    def get_report_path(self, report_type: ReportType) -> Path:
        """Find report file."""
        return self.get_output_directory() / report_type.get_file_name()

    def delete_report(self, report_type: ReportType) -> None:
        """Delete report file."""
        self.get_report_path(report_type).unlink()

    def test_report_exists(self) -> None:
        """Find HTML report file."""
        self.generate_report(self.REPORT_TYPE)
        report_path = self.get_report_path(self.REPORT_TYPE)

        assert report_path.exists()
