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

"""HTML document based report renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jinja2

from cssfinder.reports.renderer import Renderer, Report, ReportType

if TYPE_CHECKING:
    from cssfinder.cssfproject import Task
    from cssfinder.reports.math import SlopeProperties
    from cssfinder.reports.plotting import Plot


class HTMLRenderer(Renderer):
    """Renderer implementation outputting HTML files content."""

    def __init__(self, props: SlopeProperties, plots: list[Plot], task: Task) -> None:
        super().__init__(props, plots, task)
        self.env = jinja2.Environment(
            loader=jinja2.PackageLoader("cssfinder"),
            autoescape=jinja2.select_autoescape(),
        )

    def render(self) -> Report:
        """Generate report content."""
        template = self.env.get_template("report.html.jinja2")
        return Report(
            template.render(ctx=self.ctx).encode("utf-8"),
            ReportType.HTML,
            self.ctx.task.output / "report.html",
        )
