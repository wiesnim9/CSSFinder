"""This module contains implementation of Gilbert algorithm using numpy and complex128
precision."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cssfinder.algorithm.backend.base import BackendBase
from cssfinder.algorithm.backend.numpy import NumPy

from cssfinder.io.v1_0_0.asset_loader import State
from cssfinder.project.v1_0_0.cssfproject import AlgoMode, Backend, Precision


@BackendBase.register(Backend.NumPy, Precision.Complex128)
class NumPyC128(NumPy):
    """Implementation of Gilbert algorithm using numpy and complex128 precision."""

    _visibility: npt.NDArray[np.complex128]
    _intermediate: npt.NDArray[np.complex128]
    _corrections: list[tuple[int, int, float]]

    def __init__(self, initial: State, mode: AlgoMode, visibility: float) -> None:
        super().__init__(initial, mode, visibility)
        self._visibility = self.visibility * initial.state + (
            1 - self.visibility
        ) * np.identity(len(initial.state), dtype=np.complex128)
        self._intermediate = self.initial.state.copy()
        self._corrections = []

        self._aa4 = 2 * product(self._visibility, self._intermediate)
        self._aa6 = product(self._intermediate, self._intermediate)
        self._visibility_reduced = self._visibility - self._intermediate
        self._dd1 = product(self._intermediate, self._visibility_reduced)

    @property
    def state(self) -> npt.NDArray[np.complex128]:
        """Return current system state with all optimizations applied."""
        raise NotImplementedError()

    @property
    def corrections(self) -> list[tuple[int, int, float]]:
        """Return list of all corrections found during optimization."""
        raise NotImplementedError()

    @property
    def corrections_count(self) -> int:
        """Return number of all corrections found during optimization."""
        raise NotImplementedError()

    def run_epoch(self, iterations: int, epoch_index: int) -> None:
        """Run sequence of iterations without stopping to check any stop conditions."""

        depth = self.initial.depth
        quantity = self.initial.quantity
        epochs = 20 * depth * depth * quantity

        for iteration_index in range(iterations):
            alternative_state = random_d_fs(depth, quantity)

            if product(alternative_state, self._visibility_reduced) > self._dd1:
                alternative_state = optimize_d_fs(
                    alternative_state, self._visibility_reduced, depth, quantity, epochs
                )

                aa3 = product(alternative_state, alternative_state)
                aa2 = 2 * product(self._visibility, alternative_state)
                aa5 = 2 * product(self._intermediate, alternative_state)

                c = -(-self._aa4 + aa2 + aa5 - 2 * aa3) / (2 * (self._aa6 - aa5 + aa3))

                if 0 <= c <= 1:

                    self._intermediate = (
                        c * self._intermediate + (1 - c) * alternative_state
                    )

                    self._visibility_reduced = self._visibility - self._intermediate
                    self._aa4 = 2 * product(self._visibility, self._intermediate)
                    self._aa6 = product(self._intermediate, self._intermediate)
                    self._dd1 = self._aa4 / 2 - self._aa6

                    self._corrections.append(
                        (
                            epoch_index * iterations + iteration_index,
                            len(self._corrections),
                            float(
                                product(
                                    self._visibility_reduced,
                                    self._visibility_reduced,
                                )
                            ),
                        )
                    )


# def create_rho_1(prefix, rho, mode, d1, vis):
#     rhoa = np.zeros(rho.shape, dtype=np.complex128)
#     np.fill_diagonal(rhoa, rho.diagonal())
#     return rhoa


# @jit(nopython=True, nogil=True, cache=True)
def product(
    matrix1: npt.NDArray[np.complex128], matrix2: npt.NDArray[np.complex128]
) -> np.float64:
    """Calculate scalar product of two matrices."""
    return np.trace(np.dot(matrix1, matrix2)).real  # type: ignore


# @jit(nopython=True, nogil=True, cache=True)
def normalize(mtx: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Normalization of a vector."""
    mtx2 = np.dot(mtx, np.conj(mtx))
    val = np.sqrt(np.real(mtx2))
    return mtx / val  # type: ignore


# @jit(nopython=True, nogil=True, cache=True)
def get_random_haar(size: int) -> npt.NDArray[np.complex128]:
    """Generate a random vector with Haar measure."""
    real = np.random.normal(0, 1, size)
    imaginary = np.random.normal(0, 1, size)
    return real + 1j * imaginary


# @jit(nopython=True, nogil=True, cache=True)
def project(mtx1: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Build a projection from a vector."""
    return np.outer(mtx1, np.conj(mtx1))


# @jit(nopython=True, nogil=True, cache=True)
def random_d_fs(size: int, sub_sys_size: int) -> npt.NDArray[np.complex128]:
    """Random n quDit state."""
    vector = normalize(get_random_haar(size))

    for _ in range(sub_sys_size - 1):
        idx_vector = normalize(get_random_haar(size))
        vector = np.outer(vector, idx_vector).flatten()

    vector = project(vector)
    return vector  # type: ignore


# @jit(nopython=True, nogil=True, cache=True)
# @jit(forceobj=True)
def optimize_d_fs(
    rho2: npt.NDArray[np.complex128],
    rho3: npt.NDArray[np.complex128],
    size: int,
    sub_sys_size: int,
    epochs: int,
) -> npt.NDArray[np.complex128]:
    """Optimize implementation for FSnQd mode."""

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
# @jit(forceobj=True)
def random_unitary_d_fs(
    size: int, sub_sys_size: int, idx: int
) -> npt.NDArray[np.complex128]:
    """N quDits."""
    value = _random_unitary_d_fs_val(size)
    mtx = expand_d_fs(value, size, sub_sys_size, idx)
    return mtx


# @jit(nopython=True, nogil=True, cache=True)
def _random_unitary_d_fs_val(size: int) -> npt.NDArray[np.complex128]:
    real = np.cos(0.01 * np.pi)
    imag = 1j * np.sin(0.01 * np.pi)
    value = real + imag - 1

    random_mtx = random_d_fs(size, 1)
    identity_mtx = np.identity(size).astype(np.complex128)
    value = np.add(np.multiply(value, random_mtx), identity_mtx)
    return value


# @jit(nopython=True, nogil=True, cache=True)
# @jit(forceobj=True)
def expand_d_fs(  # pylint: disable=invalid-name
    value: npt.NDArray[np.complex128],
    size: int,
    sub_sys_size: int,
    idx: int,
) -> npt.NDArray[np.complex128]:
    """Expand an operator to n quDits."""
    size_1 = int(size**idx)
    identity_1 = np.identity(size_1).astype(np.complex128)

    size_2 = int(size ** (sub_sys_size - idx - 1))
    identity_2 = np.identity(size_2).astype(np.complex128)

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


# @jit(nopython=True, nogil=True, cache=True)
def rotate(
    rho2: npt.NDArray[np.complex128], unitary: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Sandwich an operator with a unitary."""
    rho2a = np.dot(rho2, np.conj(unitary).T)  # matmul replaced with dot
    rho2a = np.dot(unitary, rho2a)  # matmul replaced with dot
    return rho2a  # type: ignore
