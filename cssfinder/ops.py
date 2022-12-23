from __future__ import annotations
import numpy as np
from numba import jit

from cssfinder.types import MtxC128T


@jit(nopython=True, nogil=True, cache=True)
def product(mtx1: MtxC128T, mtx2: MtxC128T) -> float:
    """Scalar product of two matrices."""
    k = np.trace(np.dot(mtx1, mtx2))  # matmul replaced with dot
    return float(np.real(k))


@jit(nopython=True, nogil=True, cache=True)
def outer_flatten(mtx: MtxC128T, mtx2: MtxC128T) -> MtxC128T:
    """Outer product of two vectors."""
    return np.outer(mtx, mtx2).flatten()


@jit(nopython=True, nogil=True, cache=True)
def normalize(mtx: MtxC128T) -> MtxC128T:
    """Normalization of a vector."""
    mtx2 = np.dot(mtx, np.conj(mtx))
    val = np.sqrt(np.real(mtx2))
    return mtx / val  # type: ignore


@jit(nopython=True, nogil=True, cache=True)
def get_random_haar(size: int) -> MtxC128T:
    """Generate a random vector with Haar measure."""
    real = np.random.normal(0, 1, size)
    imaginary = np.random.normal(0, 1, size)
    return real + 1j * imaginary


@jit(nopython=True, nogil=True, cache=True)
def project(mtx1: MtxC128T) -> MtxC128T:
    """Build a projection from a vector."""
    return np.outer(mtx1, np.conj(mtx1))


@jit(nopython=True, nogil=True, cache=True)
def random_d_fs(size: int, sub_sys_size: int) -> MtxC128T:
    """Random n quDit state."""
    vector = normalize(get_random_haar(size))

    for _ in range(sub_sys_size - 1):
        idx_vector = normalize(get_random_haar(size))
        vector = outer_flatten(vector, idx_vector)

    vector = project(vector)
    return vector  # type: ignore


# @jit(nopython=True, nogil=True, cache=True)
def optimize_d_fs(
    rho2: MtxC128T, rho3: MtxC128T, size: int, sub_sys_size: int, epochs: int
) -> MtxC128T:
    """Optimize implementation for FSQN mode."""

    product_2_3 = product(rho2, rho3)

    # To make sure rotated_2 is not unbound
    unitary = random_unitary_d_fs(size, sub_sys_size, 0)
    rotated_2 = rotate(rho2, unitary)

    for idx in range(epochs):

        idx_mod = idx % int(sub_sys_size)
        unitary = random_unitary_d_fs(size, sub_sys_size, idx_mod)
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
def random_unitary_d_fs(size: int, sub_sys_size: int, idx: int) -> MtxC128T:
    """n quDits"""
    real = np.cos(0.01 * np.pi)
    imag = 1j * np.sin(0.01 * np.pi)
    value = real + imag - 1

    random_mtx = random_d_fs(size, 1)
    identity_mtx = np.identity(size).astype(np.complex128)
    value = np.add(np.multiply(value, random_mtx), identity_mtx)

    mtx = expand_d_fs(value, size, sub_sys_size, idx)
    return mtx


# @jit(nopython=True, nogil=True, cache=True)
def expand_d_fs(  # pylint: disable=invalid-name
    value: MtxC128T,
    size: int,
    sub_sys_size: int,
    idx: int,
) -> MtxC128T:
    """Expand an operator to n quDits."""
    size_1 = int(size**idx)
    identity_1 = np.identity(size_1).astype(np.complex128)

    size_2 = int(size ** (sub_sys_size - idx - 1))
    identity_2 = np.identity(size_2).astype(np.complex128)

    kronecker_1 = kronecker(identity_1, value)
    kronecker_2 = kronecker(kronecker_1, identity_2)

    return kronecker_2

# @jit(nopython=True, nogil=True, cache=True)
def kronecker(mtx: MtxC128T, mtx1: MtxC128T) -> MtxC128T:
    """Kronecker Product."""
    ddd1 = len(mtx)
    ddd2 = len(mtx1)

    output_shape = (ddd1 * ddd2, ddd1 * ddd2)
    tdot_0_1 = np.tensordot(mtx, mtx1, 0)
    out_mtx = np.swapaxes(tdot_0_1, 1, 2)

    return out_mtx.reshape(output_shape)


@jit(nopython=True, nogil=True, cache=True)
def rotate(rho2: MtxC128T, unitary: MtxC128T) -> MtxC128T:
    """Sandwich an operator with a unitary."""
    rho2a = np.dot(rho2, np.conj(unitary).T)  # matmul replaced with dot
    rho2a = np.dot(unitary, rho2a)  # matmul replaced with dot
    return rho2a  # type: ignore
