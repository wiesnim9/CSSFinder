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


"""Abstract base class for implementing report renderers."""


from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from cssfinder.cssfproject import Task
    from cssfinder.reports.math import SlopeProperties
    from cssfinder.reports.plotting import Plot


class ReportType(Enum):
    """Possible report types."""

    HTML = "html"
    PDF = "pdf"
    ARCHIVE = "zip"
    TXT = "txt"


class Renderer(ABC):
    """Base class for creating report renderers."""

    def __init__(self, props: SlopeProperties, plots: list[Plot], task: Task) -> None:
        self.ctx = Ctx(props, plots, task)

    @abstractmethod
    def render(self) -> Report:
        """Render report.

        Returns
        -------
        RenderedReport
            Report handle providing interface for saving report.

        """


@dataclass
class Ctx:
    """Report template rendering context."""

    props: SlopeProperties
    plots: list[Plot]
    task: Task

    @property
    def title(self) -> str:
        """Document title."""
        return f"Report {self.task.output.parent.parent.name} / {self.task.name}"

    @property
    def meta(self) -> OrderedDict:
        """Return project metadata."""
        return OrderedDict(
            {
                "Project name": self.task.project.meta.name,
                "Task name": self.task.name,
                "Author": self.task.project.meta.author,
                "Email": self.task.project.meta.email,
                "Description": self.task.project.meta.description,
                "Version": self.task.project.meta.version,
            }
        )

    @property
    def math_props(self) -> OrderedDict:
        """Return mathematical properties."""
        return OrderedDict(
            {
                "Hilbert-Schmidt distance": f"{self.props.optimum:.3f}",
                "Sample correlation coefficient": f"{self.props.r_value:.3f}",
            }
        )


class Report:
    """Container for rendered report."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    def save_to(self, dest: Path) -> None:
        """Save report to a file.

        Parameters
        ----------
        dest : Path
            Path to destination file.

        """
        dest.write_bytes(self.content)
