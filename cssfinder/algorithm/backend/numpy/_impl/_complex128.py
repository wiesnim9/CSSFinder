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
#
# ------------------------------------------------------------------------------------ #
#                                                                                      #
#   THIS FILE WAS AUTOMATICALLY GENERATED FROM TEMPLATE. DO NOT MODIFY.                #
#                                                                                      #
#   To modify this file, modify `scripts/templates/numpy.pyjinja2` and                 #
#   use `poe gen-numpy-impl` to generate python files.                                 #
#                                                                                      #
# ------------------------------------------------------------------------------------ #
#
"""Module contains implementation of backend operations in numpy.

Spec
----

- Floating precision:   np.float64
- Complex precision:    np.complex128

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt


#    █████  ██████  ███    ███ ███    ███  ██████  ███    ██
#   ██     ██    ██ ████  ████ ████  ████ ██    ██ ████   ██
#   ██     ██    ██ ██ ████ ██ ██ ████ ██ ██    ██ ██ ██  ██
#   ██     ██    ██ ██  ██  ██ ██  ██  ██ ██    ██ ██  ██ ██
#    █████  ██████  ██      ██ ██      ██  ██████  ██   ████


_REAL = np.cos(0.01 * np.pi)
_IMAG = 1j * np.sin(0.01 * np.pi)
_VALUE = (_REAL + _IMAG - 1).astype(np.complex128)


@jit(nopython=True, nogil=True, cache=True)
def product(
    matrix1: npt.NDArray[np.complex128], matrix2: npt.NDArray[np.complex128]
) -> np.float64:
    """Calculate scalar product of two matrices."""
    retval = np.trace(np.dot(matrix1, matrix2)).real

    return retval  # type: ignore[no-any-return]


@jit(nopython=True, nogil=True, cache=True)
def get_random_haar_1d(depth: int) -> npt.NDArray[np.complex128]:
    """Generate a random vector with Haar measure."""
    real = np.random.uniform(0, 1, depth)  # noqa: NPY002
    imag = np.random.uniform(0, 1, depth)  # noqa: NPY002

    retval = np.exp(2 * np.pi * 1j * real) * np.sqrt(-np.log(imag))

    return retval  # type: ignore[no-any-return]


@jit(nopython=True, nogil=True, cache=True)
def get_random_haar_2d(depth: int, quantity: int) -> npt.NDArray[np.complex128]:
    """Generate multiple random vectors with Haar measure in form of matrix."""
    real = np.random.uniform(0, 1, (quantity, depth))  # noqa: NPY002
    imag = np.random.uniform(0, 1, (quantity, depth))  # noqa: NPY002

    retval = np.exp(2 * np.pi * 1j * real) * np.sqrt(-np.log(imag))

    return retval  # type: ignore[no-any-return]


@jit(nopython=True, nogil=True, cache=True)
def normalize(mtx: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Normalize a vector."""
    mtx2 = np.dot(mtx, np.conj(mtx))

    val = np.sqrt(np.real(mtx2))

    retval = mtx / val

    return retval  # type: ignore[no-any-return]


@jit(nopython=True, nogil=True, cache=True)
def project(mtx1: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Build a projection from a vector."""
    retval = np.outer(mtx1, np.conj(mtx1))

    return retval  # type: ignore[no-any-return]


@jit(nopython=False, forceobj=True, cache=True, looplift=False)
def kronecker(
    mtx: npt.NDArray[np.complex128], mtx1: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Kronecker Product."""
    ddd1 = len(mtx)
    ddd2 = len(mtx1)

    output_shape = (ddd1 * ddd2, ddd1 * ddd2)

    dot_0_1 = np.tensordot(mtx, mtx1, 0)

    out_mtx = np.swapaxes(dot_0_1, 1, 2)

    retval = out_mtx.reshape(output_shape).astype(np.complex128, copy=False)

    return retval  # type: ignore[no-any-return]


@jit(nopython=True, nogil=True, cache=True)
def rotate(
    rho2: npt.NDArray[np.complex128], unitary: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Sandwich an operator with a unitary."""
    rho2a = np.dot(rho2, np.conj(unitary).T)  # matmul replaced with dot

    rho2a = np.dot(unitary, rho2a)  # matmul replaced with dot

    return rho2a  # type: ignore[no-any-return]


#   ██████     ███████    ███████            ███    ███     ██████     ██████     ███████   # noqa: E501
#   ██   ██    ██         ██                 ████  ████    ██    ██    ██   ██    ██        # noqa: E501
#   ██   ██    █████      ███████            ██ ████ ██    ██    ██    ██   ██    █████     # noqa: E501
#   ██   ██    ██              ██            ██  ██  ██    ██    ██    ██   ██    ██        # noqa: E501
#   ██████     ██         ███████            ██      ██     ██████     ██████     ███████   # noqa: E501


@jit(nopython=False, forceobj=True, cache=True, looplift=False)
def optimize_d_fs(
    new_state: npt.NDArray[np.complex128],
    visibility_state: npt.NDArray[np.complex128],
    depth: int,
    quantity: int,
    updates_count: int,
) -> npt.NDArray[np.complex128]:
    """Optimize implementation for FSnQd mode."""
    product_2_3 = product(new_state, visibility_state)

    # To make sure rotated_2 is not unbound
    unitary = random_unitary_d_fs(depth, quantity, 0)

    rotated_2 = rotate(new_state, unitary)

    for idx in range(updates_count):
        idx_mod = idx % int(quantity)
        unitary = random_unitary_d_fs(depth, quantity, idx_mod)

        rotated_2 = rotate(new_state, unitary)

        product_rot2_3 = product(rotated_2, visibility_state)

        if product_2_3 > product_rot2_3:
            unitary = unitary.conj().T
            rotated_2 = rotate(new_state, unitary)

        while (new_product_2_3 := product_rot2_3) > product_2_3:
            product_2_3 = new_product_2_3
            rotated_2 = rotate(rotated_2, unitary)

            product_rot2_3 = product(rotated_2, visibility_state)

    return rotated_2.astype(np.complex128, copy=False)  # type: ignore[no-any-return]


@jit(nopython=False, forceobj=True, cache=True, looplift=False)
def random_unitary_d_fs(
    depth: int, quantity: int, idx: int
) -> npt.NDArray[np.complex128]:
    """N quDits."""
    value = _random_unitary_d_fs(depth)

    mtx = expand_d_fs(value, depth, quantity, idx)

    return mtx  # type: ignore[no-any-return]


@jit(nopython=True, nogil=True, cache=True)
def _random_unitary_d_fs(depth: int) -> npt.NDArray[np.complex128]:
    random_mtx = random_d_fs(depth, 1)

    identity_mtx = np.identity(depth).astype(np.complex128)

    rand_mul = np.multiply(_VALUE, random_mtx)

    value = np.add(rand_mul, identity_mtx)

    return value  # type: ignore[no-any-return]


@jit(nopython=True, nogil=True, cache=True)
def random_d_fs(depth: int, quantity: int) -> npt.NDArray[np.complex128]:
    """Random n quDit state."""
    rand_vectors = get_random_haar_2d(depth, quantity)
    vector = normalize(rand_vectors[0])

    for i in range(quantity - 1):
        idx_vector = normalize(rand_vectors[i])

        vector = np.outer(vector, idx_vector).flatten()

    vector = project(vector)

    return vector  # type: ignore[no-any-return]


@jit(nopython=False, forceobj=True, cache=True, looplift=False)
def expand_d_fs(
    value: npt.NDArray[np.complex128],
    depth: int,
    quantity: int,
    idx: int,
) -> npt.NDArray[np.complex128]:
    """Expand an operator to n quDits."""
    depth_1 = int(depth**idx)
    identity_1 = np.identity(depth_1, dtype=np.complex128)

    depth_2 = int(depth ** (quantity - idx - 1))
    identity_2 = np.identity(depth_2, dtype=np.complex128)

    kronecker_1 = kronecker(identity_1, value)

    kronecker_2 = kronecker(kronecker_1, identity_2)

    return kronecker_2  # type: ignore[no-any-return]


#   ██████     ███████            ███    ███     ██████     ██████     ███████
#   ██   ██    ██                 ████  ████    ██    ██    ██   ██    ██
#   ██████     ███████            ██ ████ ██    ██    ██    ██   ██    █████
#   ██   ██         ██            ██  ██  ██    ██    ██    ██   ██    ██
#   ██████     ███████            ██      ██     ██████     ██████     ███████


@jit(nopython=True, nogil=True, cache=True)
def random_bs(depth: int, quantity: int) -> npt.NDArray[np.complex128]:
    """Draw random biseparable state."""
    random_vector_1 = normalize(get_random_haar_1d(depth))
    random_vector_2 = normalize(get_random_haar_1d(quantity))

    vector = np.outer(random_vector_1, random_vector_2).flatten()

    vector = project(vector)

    return vector  # type: ignore[no-any-return]


@jit(nopython=False, forceobj=True, cache=True, looplift=False)
def random_unitary_bs(depth: int, quantity: int) -> npt.NDArray[np.complex128]:
    """Draw random unitary for biseparable state."""
    random_vector = normalize(get_random_haar_1d(depth))

    random_matrix = project(random_vector)

    identity_depth = np.identity(depth).astype(np.complex128)

    identity_quantity = np.identity(quantity).astype(np.complex128)

    unitary_biseparable = _VALUE * random_matrix + identity_depth

    retval = kronecker(unitary_biseparable, identity_quantity)

    return retval  # type: ignore[no-any-return]


@jit(nopython=False, forceobj=True, cache=True, looplift=False)
def random_unitary_bs_reverse(depth: int, quantity: int) -> npt.NDArray[np.complex128]:
    """Draw random unitary for biseparable state."""
    random_vector = normalize(get_random_haar_1d(depth))

    random_matrix = project(random_vector)

    identity_depth = np.identity(depth).astype(np.complex128)

    identity_quantity = np.identity(quantity).astype(np.complex128)

    unitary_biseparable = _VALUE * random_matrix + identity_depth

    retval = kronecker(identity_quantity, unitary_biseparable)

    return retval  # type: ignore[no-any-return]


@jit(nopython=False, forceobj=True, cache=True, looplift=False)
def optimize_bs(
    new_state: npt.NDArray[np.complex128],
    visibility_state: npt.NDArray[np.complex128],
    depth: int,
    quantity: int,
    updates_count: int,
) -> npt.NDArray[np.complex128]:
    """Run the minimization algorithm to optimize the biseparable state.

    Parameters
    ----------
    new_state : npt.NDArray[np.complex128]
        Randomly drawn state to be optimized.
    visibility_state : npt.NDArray[np.complex128]
        Visibility matrix.
    depth : int
        Depth of analyzed system.
    quantity : int
        Quantity of quDits in system.
    updates_count : int
        Number of optimizer iterations to execute.

    Returns
    -------
    npt.NDArray[np.complex128]
        Optimized state.

    """
    pp1 = product(new_state, visibility_state)

    return_state = new_state.copy()

    for index in range(updates_count):
        if index % 2:
            unitary = random_unitary_bs(depth, quantity)
        else:
            unitary = random_unitary_bs_reverse(depth, quantity)

        return_state = rotate(new_state, unitary)

        if pp1 > product(return_state, visibility_state):
            unitary = unitary.conj().T
            return_state = rotate(new_state, unitary)

        pp2 = product(return_state, visibility_state)

        while pp2 > pp1:
            pp1 = pp2
            return_state = rotate(return_state, unitary)
            pp2 = product(return_state, visibility_state)

    return return_state
