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

"""Utilities for runtime report creation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd
    from typing_extensions import Self


def create_corrections_plot(corrections: pd.DataFrame) -> plt.Axes:
    """Create a plot of distance decay corrections.

    Parameters
    ----------
    corrections : pandas.DataFrame
        A DataFrame containing the distance decay corrections. The DataFrame
        should have an "index" column and a "value" column.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object for the created plot.

    Notes
    -----
    The function creates a line plot of the distance decay corrections,
    with the "index" column on the x-axis and the "value" column on the
    y-axis. The plot includes a grid and axis labels, and a title indicating
    that it shows distance decay.

    The function returns the axes object for the created plot, which can be
    further customized or saved using the methods of the matplotlib API.

    """
    plt.figure()
    axes = plt.subplot()

    axes.plot(corrections[["index"]], corrections[["value"]])
    axes.grid(visible=True)

    axes.set_xlabel("Correction index")
    axes.set_ylabel("Correction value")

    axes.set_title("Distance decay")

    return axes


def create_iteration_linear_plot(corrections: pd.DataFrame) -> plt.Axes:
    """Create a plot of iteration linear corrections.

    Parameters
    ----------
    corrections : pandas.DataFrame
        A DataFrame containing the iteration linear corrections. The DataFrame
        should have columns "iteration" and "index" containing the iteration
        number and correction index, respectively.

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
    plt.figure()
    axes = plt.subplot()

    axes.grid(visible=True)

    props = SlopeProperties.find(corrections.to_numpy())
    axes.plot(
        corrections[["iteration"]],
        props.get_correction(corrections[["index"]].to_numpy()),
    )

    return axes


def cov(
    array_1: npt.NDArray[np.float64],
    array_2: npt.NDArray[np.float64],
) -> np.float64:
    """Calculate the covariance between two arrays.

    Parameters
    ----------
    array_1 : numpy.ndarray
        The first array.
    array_2 : numpy.ndarray
        The second array.

    Returns
    -------
    float
        The covariance between the two arrays.

    Notes
    -----
    The covariance is calculated as the mean of the element-wise product of
    the deviation from the mean of `array_1` and `array_2`. In other words,
    the covariance measures how much two variables change together, and it
    is a measure of the linear relationship between them.

    """
    return np.mean(  # type: ignore[no-any-return]
        np.multiply(
            np.subtract(array_1, np.mean(array_1)),
            np.subtract(array_2, np.mean(array_2)),
        ),
    )


def trend(
    array_1: npt.NDArray[np.float64],
    array_2: npt.NDArray[np.float64],
) -> np.float64:
    """Calculate the trend between two arrays.

    Parameters
    ----------
    array_1 : numpy.ndarray
        The first array.
    array_2 : numpy.ndarray
        The second array.

    Returns
    -------
    float
        The trend between the two arrays.

    Notes
    -----
    The trend is calculated as the covariance between the logarithm of
    `array_1` and `array_2` divided by the covariance between the logarithm
    of `array_1` and itself.

    """
    l1a = np.log(array_1)
    l2a = np.log(array_2)

    return cov(l1a, l2a) / cov(l1a, l1a)


def offset(
    array_1: npt.NDArray[np.float64],
    array_2: npt.NDArray[np.float64],
) -> np.float64:
    """Calculate the offset between the two input arrays.

    Offset is based on their logarithmic means and a decay trend.

    Parameters
    ----------
    array_1 : numpy.ndarray[np.float64]
        The first input array.
    array_2 : numpy.ndarray[np.float64]
        The second input array.

    Returns
    -------
    numpy.float64
        The calculated offset between the two input arrays.

    Raises
    ------
    ValueError
        If the input arrays are empty.

    Examples
    --------
    ```
    >>> array_1 = np.array([1.0, 2.0, 3.0])
    >>> array_2 = np.array([4.0, 5.0, 6.0])
    >>> offset(array_1, array_2)
    0.01638058574365686

    ```

    """
    array_1_mean = np.mean(np.log(array_1))
    array_2_mean = np.mean(np.log(array_2))
    decay_trend = trend(array_1, array_2)

    return array_1_mean - array_2_mean * decay_trend  # type: ignore[no-any-return]


def find_correction_optimum(values: npt.NDArray[np.float64]) -> np.float64:
    """Find the optimum correction value for a given input array of values.

    Parameters
    ----------
    values : numpy.ndarray[np.float64]
        The input array of values for which to find the optimum correction.

    Returns
    -------
    numpy.float64
        The optimum correction value for the input array of values.

    """
    values = values
    upper_half = values[len(values) // 2 :]

    optimum = upper_half[-1] - 1e-6
    step1 = optimum / 10000

    while R(upper_half, optimum - step1) > R(upper_half, optimum) and optimum > 0:
        optimum = optimum - step1

    return optimum  # type: ignore[no-any-return]


@lru_cache(16)
def _r_indexes(length: int) -> npt.NDArray[np.float64]:
    indexes = np.arange(length)
    difference = np.mean(np.square(indexes)) - np.square(np.mean(indexes))
    return difference  # type: ignore[no-any-return]


@lru_cache(16)
def _indexes(length: int) -> npt.NDArray[np.float64]:
    return np.arange(length, dtype=np.float64)


def R(values: npt.NDArray[np.float64], a: np.float64) -> np.float64:  # noqa: N802
    """Calculate the R value for a given input array of values and a correction factor.

    Parameters
    ----------
    values : numpy.ndarray[np.float64]
        The input array of values for which to calculate the R value.
    a : numpy.float64
        The correction factor to use when calculating the R value.

    Returns
    -------
    numpy.float64
        The R value for the input array of values and correction factor.

    Examples
    --------
    >>> values = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    >>> R(values, 1.0)
    -0.964557517874

    """
    ll1 = np.divide(1.0, np.subtract(values, a))
    length = len(values)
    indexes = _indexes(length)

    aa1: np.float64 = np.mean(np.multiply(ll1, indexes)) - np.mean(ll1) * np.mean(
        indexes,
    )
    aa2: np.float64 = np.sqrt(
        (np.mean(np.square(ll1)) - np.square(np.mean(ll1))) * _r_indexes(length),
    )

    return np.divide(aa1, aa2, dtype=np.float64)  # type: ignore[no-any-return]


def display_short_report(data: npt.NDArray[np.float64]) -> None:
    """Display short report."""
    slope_properties = SlopeProperties.find(data)

    expr = f"corrections = trail ^ {slope_properties.aa1} * {slope_properties.bb1}"

    sys.stdout.write(
        "Basing on decay, the squared HS distance is estimated to be "
        f"{slope_properties.optimum} (R={slope_properties.r_value})\n",
    )
    sys.stdout.write(
        f"The dependence between correction and trail is approximately: {expr}\n",
    )


@dataclass
class SlopeProperties:
    """Class that encapsulates slope properties and provides methods to calculate
    correction values and find slope properties for a given dataset.

    Attributes
    ----------
    optimum: np.float64
        The optimum value found during the slope property calculation.
    r_value: np.float64
        The r-value calculated for the slope properties.
    aa1: np.float64
        The slope of the trend line of the correction index with respect to iteration
        index.
    bb1: np.float64
        The exponential decay coefficient calculated for the slope properties.

    Methods
    -------
    get_correction(x: npt.NDArray[np.float64]) -> np.float64:
        Returns the correction values for a given input array `x`.
    find(data: npt.NDArray[np.float64]) -> 'SlopeProperties':
        Finds the slope properties for a given dataset `data`.

    """

    optimum: np.float64
    r_value: np.float64
    aa1: np.float64
    bb1: np.float64

    def get_correction(self, x: np.float64) -> np.float64:
        """Return the correction values for a given input array `x`.

        Parameters
        ----------
        x: npt.NDArray[np.float64]
            Input array for which correction values will be calculated.

        Returns
        -------
        np.float64
            The correction values calculated for the input array `x`.

        """
        return np.multiply(  # type: ignore[no-any-return]
            np.power(x, self.aa1),
            self.bb1,
        )

    @classmethod
    def find(cls, data: npt.NDArray[np.float64]) -> Self:
        """Find the slope properties for a given dataset `data`.

        Parameters
        ----------
        data: npt.NDArray[np.float64]
            The dataset for which slope properties will be calculated.

        Returns
        -------
        SlopeProperties
            An instance of the SlopeProperties class representing the slope properties
            of the input data.

        """
        iteration_index: npt.NDArray[np.float64] = data[:, 0]
        correction_index: npt.NDArray[np.float64] = data[:, 1]
        correction_value: npt.NDArray[np.float64] = data[int(2 * len(data) / 3) :, 2]

        optimum = find_correction_optimum(data[:, 2])

        r_value = R(correction_value, optimum)

        aa1 = trend(iteration_index, correction_index)
        bb1 = np.exp(offset(iteration_index, correction_index))

        return cls(optimum, r_value, aa1, bb1)
