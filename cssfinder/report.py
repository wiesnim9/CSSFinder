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

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from typing_extensions import Self


def create_corrections_plot(corrections: pd.DataFrame) -> plt.Axes:
    """Create corrections plot from data in DataFrame.

    DataFrame object must have 2 columns: "iteration" and "value".
    """
    plt.figure()
    axes = plt.subplot()

    axes.plot(corrections[["index"]], corrections[["value"]])
    axes.grid(True)

    axes.set_xlabel("Correction index")
    axes.set_ylabel("Correction value")

    axes.set_title("Distance decay")

    return axes


def create_iteration_linear_plot(corrections: pd.DataFrame) -> plt.Axes:
    plt.figure()
    axes = plt.subplot()

    axes.grid(True)

    props = SlopeProperties.find(corrections.to_numpy())
    axes.plot(
        corrections[["iteration"]],
        props.get_correction(corrections[["index"]].to_numpy()),
    )

    return axes


def cov(
    array_1: npt.NDArray[np.float64], array_2: npt.NDArray[np.float64]
) -> np.float64:
    return np.mean(
        np.multiply(
            np.subtract(array_1, np.mean(array_1)),
            np.subtract(array_2, np.mean(array_2)),
        )
    )


def trend(
    array_1: npt.NDArray[np.float64], array_2: npt.NDArray[np.float64]
) -> np.float64:
    l1a = np.log(array_1)
    l2a = np.log(array_2)

    return cov(l1a, l2a) / cov(l1a, l1a)


def offset(
    array_1: npt.NDArray[np.float64], array_2: npt.NDArray[np.float64]
) -> np.float64:
    array_1_mean = np.mean(np.log(array_1))
    array_2_mean = np.mean(np.log(array_2))
    decay_trend = trend(array_1, array_2)
    return_value = array_1_mean - array_2_mean * decay_trend
    return return_value


def find_correction_optimum(values: npt.NDArray[np.float64]) -> np.float64:
    values = values
    upper_half = values[len(values) // 2 :]

    optimum = upper_half[-1] - 1e-6
    step1 = optimum / 10000

    while R(upper_half, optimum - step1) > R(upper_half, optimum) and optimum > 0:
        optimum = optimum - step1

    return optimum


@lru_cache(16)
def _r_indexes(length: int) -> npt.NDArray[np.float64]:
    indexes = np.arange(length)
    return np.mean(np.square(indexes)) - np.square(np.mean(indexes))


@lru_cache(16)
def _indexes(length: int) -> npt.NDArray[np.float64]:
    return np.arange(length, dtype=np.float64)


def R(values: npt.NDArray[np.float64], a: np.float64) -> np.float64:
    ll1 = np.divide(1.0, np.subtract(values, a))
    length = len(values)
    indexes = _indexes(length)

    aa1 = np.mean(np.multiply(ll1, indexes)) - np.mean(ll1) * np.mean(indexes)
    aa2 = np.sqrt(
        (np.mean(np.square(ll1)) - np.square(np.mean(ll1))) * _r_indexes(length)
    )

    return np.divide(aa1, aa2)


def display_short_report(data: npt.NDArray[np.float64]):
    slope_properties = SlopeProperties.find(data)

    expr = f"corrections = trail ^ {slope_properties.aa1} * {slope_properties.bb1}"

    print(
        "Basing on decay, the squared HS distance is estimated to be",
        f"{slope_properties.optimum} (R={slope_properties.r_value})",
    )
    print(f"The dependence between correction and trail is approximately: {expr}")


@dataclass
class SlopeProperties:
    optimum: np.float64
    r_value: np.float64
    aa1: np.float64
    bb1: np.float64

    def get_correction(self, x: np.float64) -> np.float64:
        return np.multiply(np.power(x, self.aa1), self.bb1)

    @classmethod
    def find(cls, data: npt.NDArray[np.float64]) -> Self:
        iteration_index: npt.NDArray[np.float64] = data[:, 0]
        correction_index: npt.NDArray[np.float64] = data[:, 1]
        correction_value: npt.NDArray[np.float64] = data[int(2 * len(data) / 3) :, 2]

        optimum = find_correction_optimum(data[:, 2])

        r_value = R(correction_value, optimum)

        aa1 = trend(iteration_index, correction_index)
        bb1 = np.exp(offset(iteration_index, correction_index))

        return cls(optimum, r_value, aa1, bb1)
