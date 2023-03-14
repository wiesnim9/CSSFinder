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


"""Gilbert modes related tools."""


from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cssfinder.constants import PRIMES
from cssfinder.cssfproject import AlgoMode

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass
class Dimensions:
    """Container for system dimensions."""

    depth: int
    """Depth of system, ie.

    number of dimensions in qu(D)it. (d)

    """

    quantity: int
    """Quantity of systems.

    ie. number of qu(D)its in state. (n)

    """


class ModeUtil(ABC):
    """Base class for implementing mode specific utilities eg.

    shape deduction.

    """

    @staticmethod
    def new(mode: AlgoMode) -> ModeUtil:
        """Create new ModeUtil subclass instance.

        Parameters
        ----------
        mode : AlgoMode
            Util set selector.

        Returns
        -------
        ModeUtil
            Instance of subclass of ModeUtil.

        Raises
        ------
        NotImplementedError
            For unsupported AlgoModes and incorrect input.

        """
        if mode == AlgoMode.FSnQd:
            return FSnQdUtil()

        if mode == AlgoMode.SBiPa:
            return SBiPaUtil()

        msg = f"Unsupported mode {mode.name}"
        raise NotImplementedError(msg)

    def get_dimensions(self, state: npt.NDArray[np.complex128]) -> Dimensions:
        """Detect both system depth and system quantity.

        Parameters
        ----------
        state : int
            State matrix to detect dimensions for.

        Returns
        -------
        Dimensions
            System dimensions within dedicated container.

        Raises
        ------
        ValueError
            When depth and quantity can't be determined.

        """
        return self.detect_depth_and_quantity(len(state))

    @abstractmethod
    def detect_depth_and_quantity(self, total: int) -> Dimensions:
        """Detect both system depth and system quantity.

        Parameters
        ----------
        total : int
            Dimension along one of axes. Matrix is expected to be square.

        Returns
        -------
        Dimensions
            System dimensions within dedicated container.

        Raises
        ------
        ValueError
            When depth and quantity can't be determined.

        """


class FSnQdUtil(ModeUtil):
    """FSnQd specific implementation of utilities eg.

    shape deduction.

    """

    def detect_depth_and_quantity(self, total: int) -> Dimensions:
        """Detect both system depth and system quantity.

        Parameters
        ----------
        total : int
            Dimension along one of axes. Matrix is expected to be square.

        Returns
        -------
        Dimensions
            System dimensions within dedicated container.

        Raises
        ------
        ValueError
            When depth and quantity can't be determined.

        """
        for depth in PRIMES:
            quantity = int(math.log(total, depth))

            if quantity == int(quantity):
                logging.debug(
                    "Deduced quantity %r and depth %r when given total size %r",
                    depth,
                    quantity,
                    total,
                )
                return Dimensions(depth, quantity)

        reason = "prime number range exceeded"
        raise UndefinedSystemSizeError(reason)


class SBiPaUtil(ModeUtil):
    """SBiPa specific implementation of utilities eg.

    shape deduction.

    """

    def detect_depth_and_quantity(self, total: int) -> Dimensions:
        """Detect both system depth and system quantity.

        Parameters
        ----------
        total : int
            Dimension along one of axes. Matrix is expected to be square.

        Returns
        -------
        Dimensions
            System dimensions within dedicated container.

        Raises
        ------
        ValueError
            When depth and quantity can't be determined.

        """
        total_sqrt = math.sqrt(total)
        floored_total_sqrt = int(total_sqrt)

        if total_sqrt == floored_total_sqrt:
            return Dimensions(floored_total_sqrt, floored_total_sqrt)

        for depth in PRIMES:
            if total % depth == 0:
                quantity = int(total / depth)
                logging.debug(
                    "Deduced quantity %r and depth %r when given total size %r",
                    depth,
                    quantity,
                    total,
                )
                return Dimensions(depth, quantity)

        reason = "prime number range exceeded"
        raise UndefinedSystemSizeError(reason)


class UndefinedSystemSizeError(ValueError):
    """Raised when it is not possible to determine system dimensions."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Couldn't determine size of system: {reason}.")
