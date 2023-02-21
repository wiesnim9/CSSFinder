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


"""This module contains implementation of Gilbert algorithm using numpy and complex128
precision."""
from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def product(
    matrix1: npt.NDArray[np.complex128], matrix2: npt.NDArray[np.complex128]
) -> np.float64:
    """Calculate scalar product of two matrices."""
    return np.trace(np.dot(matrix1, matrix2)).real  # type: ignore


@jit(nopython=True, nogil=True, cache=True)
def normalize(mtx: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Normalization of a vector."""
    mtx2 = np.dot(mtx, np.conj(mtx))
    val = np.sqrt(np.real(mtx2))
    return mtx / val  # type: ignore


@jit(nopython=True, nogil=True, cache=True)
def get_random_haar(depth: int) -> npt.NDArray[np.complex128]:
    """Generate a random vector with Haar measure."""
    real = np.random.normal(0, 1, depth)
    imaginary = np.random.normal(0, 1, depth)
    return real + 1j * imaginary


@jit(nopython=True, nogil=True, cache=True)
def project(mtx1: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Build a projection from a vector."""
    return np.outer(mtx1, np.conj(mtx1))


@jit(nopython=True, nogil=True, cache=True)
def random_d_fs(depth: int, quantity: int) -> npt.NDArray[np.complex128]:
    """Random n quDit state."""
    vector = normalize(get_random_haar(depth))

    for _ in range(quantity - 1):
        idx_vector = normalize(get_random_haar(depth))
        vector = np.outer(vector, idx_vector).flatten()

    vector = project(vector)
    return vector  # type: ignore


# @jit(nopython=True, nogil=True, cache=True)
# @jit(forceobj=True)
def optimize_d_fs(
    rho2: npt.NDArray[np.complex128],
    rho3: npt.NDArray[np.complex128],
    depth: int,
    quantity: int,
    epochs: int,
) -> npt.NDArray[np.complex128]:
    """Optimize implementation for FSnQd mode."""

    product_2_3 = product(rho2, rho3)

    # To make sure rotated_2 is not unbound
    unitary = random_unitary_d_fs(depth, quantity, 0)
    rotated_2 = rotate(rho2, unitary)

    for idx in range(epochs):
        idx_mod = idx % int(quantity)
        unitary = random_unitary_d_fs(depth, quantity, idx_mod)
        rotated_2 = rotate(rho2, unitary)

        product_rot2_3 = product(rotated_2, rho3)

        if product_2_3 > product_rot2_3:
            unitary = unitary.conj().T
            rotated_2 = rotate(rho2, unitary)

        while (new_product_2_3 := product_rot2_3) > product_2_3:
            product_2_3 = new_product_2_3
            rotated_2 = rotate(rotated_2, unitary)
            product_rot2_3 = product(rotated_2, rho3)

    return rotated_2


# @jit(nopython=True, nogil=True, cache=True)
# @jit(forceobj=True)
def random_unitary_d_fs(
    depth: int, quantity: int, idx: int
) -> npt.NDArray[np.complex128]:
    """N quDits."""
    value = _random_unitary_d_fs_val(depth)
    mtx = expand_d_fs(value, depth, quantity, idx)
    return mtx


@jit(nopython=True, nogil=True, cache=True)
def _random_unitary_d_fs_val(depth: int) -> npt.NDArray[np.complex128]:
    real = np.cos(0.01 * np.pi)
    imag = 1j * np.sin(0.01 * np.pi)
    value = real + imag - 1

    random_mtx = random_d_fs(depth, 1)
    identity_mtx = np.identity(depth).astype(np.complex128)
    value = np.add(np.multiply(value, random_mtx), identity_mtx)
    return value


# @jit(nopython=True, nogil=True, cache=True)
# @jit(forceobj=True)
def expand_d_fs(
    value: npt.NDArray[np.complex128],
    depth: int,
    quantity: int,
    idx: int,
) -> npt.NDArray[np.complex128]:
    """Expand an operator to n quDits."""
    depth_1 = int(depth**idx)
    identity_1 = np.identity(depth_1).astype(np.complex128)

    depth_2 = int(depth ** (quantity - idx - 1))
    identity_2 = np.identity(depth_2).astype(np.complex128)

    kronecker_1 = kronecker(identity_1, value)
    kronecker_2 = kronecker(kronecker_1, identity_2)

    return cast(npt.NDArray[np.complex128], kronecker_2)


# @jit(nopython=True, nogil=True, cache=True)
# @jit(forceobj=True)
def kronecker(
    mtx: npt.NDArray[np.complex128], mtx1: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Kronecker Product."""
    ddd1 = len(mtx)
    ddd2 = len(mtx1)

    output_shape = (ddd1 * ddd2, ddd1 * ddd2)
    dot_0_1 = np.tensordot(mtx, mtx1, 0)
    out_mtx = np.swapaxes(dot_0_1, 1, 2)

    return out_mtx.reshape(output_shape)


@jit(nopython=True, nogil=True, cache=True)
def rotate(
    rho2: npt.NDArray[np.complex128], unitary: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Sandwich an operator with a unitary."""
    rho2a = np.dot(rho2, np.conj(unitary).T)  # matmul replaced with dot
    rho2a = np.dot(unitary, rho2a)  # matmul replaced with dot
    return rho2a  # type: ignore
