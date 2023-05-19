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


"""Utilities for plot creation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from matplotlib import figure as fig
from matplotlib import pyplot as plt

from cssfinder.reports.math import SlopeProperties

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self


class Plotter:
    """Plot creator class."""

    def __init__(self, corrections: pd.DataFrame) -> None:
        """Initialize plot creator.

        Parameters
        ----------
        corrections : pandas.DataFrame
            A DataFrame containing the distance decay corrections. The DataFrame
            should have an "index" column and a "value" column.

        """
        self.corrections = corrections
        self.slope_props = SlopeProperties.find(self.corrections.to_numpy())

    def plot_corrections(self, axes: Optional[plt.Axes] = None) -> Plot:
        """Create a plot of distance decay corrections.

        Parameters
        ----------
        axes : Optional[plt.Axes], optional
            Optional axes object to reuse, when none is given, new figure is created,
            by default None

        Returns
        -------
        Plot
            Plot object containing plot axes.

        Notes
        -----
        The function creates a line plot of the distance decay corrections,
        with the "index" column on the x-axis and the "value" column on the
        y-axis. The plot includes a grid and axis labels, and a title indicating
        that it shows distance decay.

        The function returns the Plot object granting access to axes for the created
        plot, which can be further customized or saved using the methods of the
        matplotlib API.

        """
        if axes is None:
            plt.figure()
            axes = plt.subplot()

        axes.plot(
            self.corrections[["index"]], self.corrections[["value"]], label="correction"
        )
        axes.hlines(
            [self.slope_props.optimum],
            xmin=-10,
            xmax=self.corrections[["index"]].max(),
            color="red",
            label="H-S distance",
            linestyles="dashed",
        )
        axes.grid(visible=True)

        axes.set_xlabel("Correction index")
        axes.set_ylabel("Correction value")

        axes.set_title("Distance decay")
        plt.legend(loc="upper right")

        return Plot(axes)

    def plot_corrections_inverse(self, axes: Optional[plt.Axes] = None) -> Plot:
        """Create a plot offsets inverse of distance decay corrections.

        Parameters
        ----------
        axes : Optional[plt.Axes], optional
            Optional axes object to reuse, when none is given, new figure is created,
            by default None

        Returns
        -------
        Plot
            Plot object containing plot axes.

        Notes
        -----
        The function creates a line plot of the inverse of distance decay corrections,
        with the "index" column on the x-axis and the "value" column inverse on the
        y-axis. The plot includes a grid and axis labels, and a title indicating
        that it shows distance decay.

        The function returns the Plot object granting access to axes for the created
        plot, which can be further customized or saved using the methods of the
        matplotlib API.

        """
        if axes is None:
            plt.figure()
            axes = plt.subplot()

        axes.plot(
            self.corrections[["index"]],
            1 / (self.corrections[["value"]] - self.slope_props.optimum),
        )
        axes.grid(visible=True)

        axes.set_xlabel("Correction index")
        axes.set_ylabel("Inverse correction value with offset")

        axes.set_title("Inverse distance decay with offset")

        return Plot(axes)

    def plot_iteration(self, axes: Optional[plt.Axes] = None) -> Plot:
        """Create a plot of iteration linear corrections.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object for the created plot.

        Notes
        -----
        The function creates a line plot of the iteration linear corrections,
        with the "iteration" column on the x-axis and the correction values on
        the y-axis. The correction values are calculated using the
        `SlopeProperties` class, which takes the "index" column as input and
        returns the corresponding correction values for each iteration.

        The plot includes a grid and axis labels, but no title. The function returns
        the axes object for the created plot, which can be further customized or
        saved using the methods of the matplotlib API.

        """
        if axes is None:
            plt.figure()
            axes = plt.subplot()

        axes.grid(visible=True)

        axes.set_xlabel("Iteration index")
        axes.set_ylabel("Correction index")

        axes.set_title("Total number of correction")

        axes.plot(
            self.corrections[["iteration"]],
            self.corrections[["index"]],
        )

        return Plot(axes)


@dataclass
class Plot:
    """Container class for plots generated with Plotter class."""

    axes: plt.Axes

    @property
    def figure(self) -> fig.Figure:
        """Axes figure."""
        return self.axes.figure

    def configure(self, width: int = 8, height: int = 6) -> Self:
        """Set the size of the current figure.

        Parameters
        ----------
        width : int, optional
            The width of the figure in inches. Default is 10.
        height : int, optional
            The height of the figure in inches. Default is 10.

        Returns
        -------
        Self
            Returns the instance of the object to allow for method chaining.

        """
        self.axes.figure.set_figwidth(width)
        self.axes.figure.set_figheight(height)
        return self

    def save_plot(
        self,
        dest: Path | BytesIO,
        dpi: int = 300,
        file_format: Optional[str] = None,
    ) -> None:
        """Save figure to file.

        Parameters
        ----------
        dest : Path | BytesIO
            Path to file or writable BytesIO.
        dpi : int, optional
            Plot output dpi, by default 150
        file_format : Optional[str], optional
            File format, when None, deduced from file path, by default None

        """
        self.axes.figure.savefig(
            dest.as_posix() if isinstance(dest, Path) else dest,
            dpi=dpi,
            format=file_format,
        )

    def base64_encode(self, file_format: Optional[str] = None) -> str:
        """Encode plot as base64 string.

        Parameters
        ----------
        file_format : Optional[str], optional
            Preferred file format, by default None

        Returns
        -------
        str
            Encoded image.

        """
        io = BytesIO()
        self.save_plot(io, file_format=file_format)
        io.seek(0)

        return base64.b64encode(io.read()).decode("utf-8")
