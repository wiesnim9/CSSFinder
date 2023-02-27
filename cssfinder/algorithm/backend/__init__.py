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


"""Backend is an implementation of Gilbert algorithm implemented with specific tools and
supporting various precisions of operation."""


from __future__ import annotations

from typing import Type

from cssfinder.algorithm.backend.base import BackendBase
from cssfinder.algorithm.backend.numpy.complex64 import NumPyC64
from cssfinder.algorithm.backend.numpy.complex128 import NumPyC128
from cssfinder.project.cssfproject import Backend, Precision


def new(backend: Backend, precision: Precision) -> Type[BackendBase]:
    """Select one of the backends with fixed precision."""
    if backend == Backend.NumPy:
        if precision == Precision.DOUBLE:
            return NumPyC128
        if precision == Precision.FULL:
            return NumPyC64

    raise UnsupportedBackendError(
        f"Backend {backend.name!r} with precision {precision.name!r} not supported."
    )


class UnsupportedBackendError(Exception):
    """Raised for unsupported backend type."""