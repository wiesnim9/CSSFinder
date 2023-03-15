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

"""Implementation of class wrapper around report generation process."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Type

from cssfinder.io.gilbert_io import GilbertIO
from cssfinder.reports.plotting import Plot, Plotter
from cssfinder.reports.renderer import Renderer, Report, ReportType

if TYPE_CHECKING:
    import pandas as pd

    from cssfinder.cssfproject import CSSFProject, Task
    from cssfinder.reports.math import SlopeProperties


class ReportManager:
    """Class wrapper around report generation process."""

    def __init__(self, project: CSSFProject, task: Task) -> None:
        """Initialize a ReportManager object with a CSSFProject and Task.

        Parameters
        ----------
        project : CSSFProject
            The CSSFProject object associated with the task.
        task : Task
            The Task object for which report will be generated.

        """
        self.project = project
        self.task = task
        self.loader = GilbertIO()

    def prepare(self) -> PreparedReportManager:
        """Prepare the data for generating a report.

        This method loads corrections from a Gilbert output file, generates plots, and
        returns a PreparedReportManager object that can be used to generate a report.

        Returns
        -------
        PreparedReportManager
            A PreparedReportManager object that contains the data and plots needed for
            generating a report.

        """
        corrections = self.loader.load_corrections(self.task.output_corrections_file)

        plots = OrderedDict()

        plotter = Plotter(corrections)

        plots["decay"] = plotter.plot_corrections()
        plots["inverse_decay"] = plotter.plot_corrections_inverse()
        plots["iterations"] = plotter.plot_iteration()

        return PreparedReportManager(
            self.project, plotter.slope_props, plots, self.task, corrections
        )


@dataclass
class PreparedReportManager:
    """Report manager with calculated report values."""

    project: CSSFProject
    props: SlopeProperties
    plots: OrderedDict[str, Plot]
    task: Task
    corrections: pd.DataFrame

    RENDERERS: ClassVar[dict[ReportType, Type[Renderer]]] = {}

    @classmethod
    def register_renderer(
        cls, renderer_cls: Type[Renderer], report_type: ReportType
    ) -> None:
        """Register renderer for report type."""
        cls.RENDERERS[report_type] = renderer_cls

    def request_report(self, report_type: ReportType | Any) -> Report:
        """Generate report."""
        renderer_cls = self.RENDERERS.get(report_type)

        if renderer_cls is None:
            if isinstance(report_type, ReportType):
                msg = f"Report type {report_type.name} is not supported yet."
                raise NotImplementedError(msg)

            msg = f"Unknown report type {report_type!r}"
            raise KeyError(msg)

        return renderer_cls(
            self.props,
            [p.configure() for p in self.plots.values()],
            self.task,
        ).render()
