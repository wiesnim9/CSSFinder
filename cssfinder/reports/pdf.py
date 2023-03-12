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


"""Renderer implementation outputting PDF file."""


from __future__ import annotations

import logging
import os
from platform import system
from typing import Any

from cssfinder.reports.html import HTMLRenderer
from cssfinder.reports.renderer import Report, ReportType

WEASYPRINT_NOT_AVAILABLE = (
    "CSSFinder failed to load PDF rendering backend. Therefore PDF reports are "
    "unavailable. To overcome this issue, visit `weasyprint` documentation "
    "installation guidelines on "
    "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation "
    "and follow instructions for you platform ("
    f"{system() if system() != 'Darwin' else 'macOS'}). "
)

if system() == "Windows":
    os.add_dll_directory(  # type: ignore[attr-defined]
        os.environ.get("WEASYPRINT_DLL_DIRECTORIES", "C:/tools/msys64/mingw64/bin")
    )


class WEasyPrintNotAvailableError(Exception):
    """Error raised on Mac OS when attempting to generate PDF report."""

    def __init__(self) -> None:
        super().__init__(WEASYPRINT_NOT_AVAILABLE)


try:
    import weasyprint

except (ImportError, OSError) as exc:
    WEASYPRINT_NOT_AVAILABLE += (
        f"\n\nError details:\n\n{exc.__class__.__qualname__}: {exc}"
    )
    logging.warning(WEASYPRINT_NOT_AVAILABLE)

    class weasyprint:  # type: ignore[no-redef] # noqa: N801
        """Dummy class for Mac OS where weasyprint fails to import."""

        def HTML(*_a: Any, **__kw: Any) -> Any:  # noqa: N802
            """Raise exception on Mac OS."""
            raise WEasyPrintNotAvailableError


class PDFRenderer(HTMLRenderer):
    """Renderer implementation outputting PDF files content."""

    def render(self) -> Report:
        """Generate report content."""
        report = super().render()
        return Report(
            weasyprint.HTML(string=report.content.decode("utf-8")).write_pdf(),
            ReportType.PDF,
            self.ctx.task.output / "report.pdf",
        )
