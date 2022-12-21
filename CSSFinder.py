# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 07:55:58 2022
ver 1.1.0
@author: wiesn
"""
from __future__ import annotations
from io import StringIO
from pathlib import Path

import sys
from typing import Literal, TypeAlias, TypeVar
import numpy as np
import math
from numpy.typing import NDArray

# import scipy.io
from datetime import datetime
from os.path import exists

# fmt: off
PRIMES = (
    2,      3,      5,      7,      11,     13,     17,     19,     23,     29,     31,     37,
    41,     43,     47,     53,     59,     61,     67,     71,     73,     79,     83,     89,
    97,     101,    103,    107,    109,    113,    127,    131,    137,    139,    149,    151,
    157,    163,    167,    173,    179,    181,    191,    193,    197,    199,    211,    223,
    227,    229,    233,    239,    241,    251,    257,    263,    269,    271,    277,    281,
    283,    293,    307,    311,    313,    317,    331,    337,    347,    349,    353,    359,
    367,    373,    379,    383,    389,    397,    401,    409,    419,    421,    431,    433,
    439,    443,    449,    457,    461,    463,    467,    479,    487,    491,    499,    503,
    509,    521,    523,    541,    547,    557,    563,    569,    571,    577,    587,    593,
    599,    601,    607,    613,    617,    619,    631,    641,    643,    647,    653,    659,
    661,    673,    677,    683,    691,    701,    709,    719,    727,    733,    739,    743,
    751,    757,    761,    769,    773,    787,    797,    809,    811,    821,    823,    827,
    829,    839,    853,    857,    859,    863,    877,    881,    883,    887,    907,    911,
    919,    929,    937,    941,    947,    953,    967,    971,    977,    983,    991,    997,
    1009,   1013,   1019,   1021,   1031,   1033,   1039,   1049,   1051,   1061,   1063,   1069,
    1087,   1091,   1093,   1097,   1103,   1109,   1117,   1123,   1129,   1151,   1153,   1163,
    1171,   1181,   1187,   1193,   1201,   1213,   1217,   1223,   1229,   1231,   1237,   1249,
    1259,   1277,   1279,   1283,   1289,   1291,   1297,   1301,   1303,   1307,   1319,   1321,
    1327,   1361,   1367,   1373,   1381,   1399,   1409,   1423,   1427,   1429,   1433,   1439,
    1447,   1451,   1453,   1459,   1471,   1481,   1483,   1487,   1489,   1493,   1499,   1511,
    1523,   1531,   1543,   1549,   1553,   1559,   1567,   1571,   1579,   1583,   1597,   1601,
    1607,   1609,   1613,   1619,   1621,   1627,   1637,   1657,   1663,   1667,   1669,   1693,
    1697,   1699,   1709,   1721,   1723,   1733,   1741,   1747,   1753,   1759,   1777,   1783,
    1787,   1789,   1801,   1811,   1823,   1831,   1847,   1861,   1867,   1871,   1873,   1877,
    1879,   1889,   1901,   1907,   1913,   1931,   1933,   1949,   1951,   1973,   1979,   1987,
    1993,   1997,   1999,
)
# fmt: on

DTYPE: TypeAlias = np.complex64


class MtxFn:
    """Matrix functions."""

    @staticmethod
    def outer(p_1: NDArray[DTYPE], p_2: NDArray[DTYPE]) -> NDArray[DTYPE]:
        """Outer product of two vectors."""
        return np.outer(np.array(p_1), np.array(p_2)).flatten()

    @staticmethod
    def normalize(p: NDArray[DTYPE]) -> NDArray[DTYPE]:
        """Normalization of a vector."""
        p_2 = np.dot(p, np.conj(p))
        p_2 = math.sqrt(np.real(p_2))
        return p / p_2

    @staticmethod
    def project(p_1: NDArray[DTYPE]) -> NDArray[DTYPE]:
        """Build a projection from a vector."""
        return np.outer(np.array(p_1), np.conj(np.array(p_1)))

    @staticmethod
    def product(mtx1: NDArray[DTYPE], mtx2: NDArray[DTYPE]):
        """Scalar product of two matrices/"""
        k = np.trace(np.matmul(mtx1, mtx2))
        return np.real(k)

    @staticmethod
    def generate(size: int | tuple[int, ...]) -> NDArray[np.complex64]:
        """Generate a random vector with Haar measure."""
        k_1 = np.random.normal(0, 1, size).astype(np.complex64)
        k_2 = np.random.normal(0, 1, size).astype(np.complex64)
        k_3 = k_1 + complex(0, 1) * k_2
        return k_3

    @staticmethod
    def ident_mtx(size: int) -> NDArray[np.complex64]:
        """Return identity matrix"""
        p_1 = np.zeros((int(size), int(size)), dtype=np.complex64)
        np.fill_diagonal(p_1, 1)
        return p_1

    @staticmethod
    def kronecker(p_1: NDArray[DTYPE], p_2: NDArray[DTYPE]) -> NDArray[DTYPE]:
        """Kronecker Product."""
        ddd1 = len(p_1)
        ddd2 = len(p_2)
        p_3 = np.reshape(
            np.swapaxes(np.tensordot(p_1, p_2, 0), 1, 2),
            (ddd1 * ddd2, ddd1 * ddd2),
        )
        return p_3

    @staticmethod
    def expand2FS(  # pylint: disable=invalid-name
        p_1: NDArray[DTYPE], n: int, j1: int
    ) -> NDArray[DTYPE]:
        """Expand an operator to n qubits."""
        return MtxFn.kronecker(
            MtxFn.kronecker(MtxFn.ident_mtx(2**j1), p_1),
            MtxFn.ident_mtx(2 ** (n - j1 - 1)),
        )

    @staticmethod
    def expand_d_FS(  # pylint: disable=invalid-name
        p_1: NDArray[DTYPE],
        d: int,
        n: int,
        j_1: int,
    ) -> NDArray[DTYPE]:
        """Expand an operator to n qubits."""
        return MtxFn.kronecker(
            MtxFn.kronecker(MtxFn.ident_mtx(int(d**j_1)), p_1),
            MtxFn.ident_mtx(int(d ** (n - j_1 - 1))),
        )

    @staticmethod
    def rotate(rho2: NDArray[DTYPE], U: NDArray[DTYPE]) -> NDArray[DTYPE]:
        """Sandwich an operator with a unitary."""
        rho2a = np.matmul(rho2, np.conj(U).T)
        rho2a = np.matmul(U, rho2a)
        return rho2a


# MTX file read and write
# [1]: 'matrix' 'vector'
# [2]: 'array' 'coordinate'
# [3]: 'integer' 'real' 'complex' 'pattern'
# [4]: 'general' 'symmetric' 'skew-symmetric' 'hermitian


class MtxIO:
    """From and to file matrix serialization."""

    @staticmethod
    def get_number(
        file: StringIO, vartype: Literal[0] | Literal[1] | Literal[2]
    ) -> int | float | complex:

        readnum = file.readline()

        if vartype == 0:
            return int(readnum)

        if vartype == 1:
            return float(readnum)

        if vartype == 2:
            readnum = readnum.split()
            kreadnum = list(map(float, readnum))
            return kreadnum[0] + complex(0, 1) * kreadnum[1]

        raise AssertionError()

    @staticmethod
    def get_3_numbers(
        file: StringIO, vartype: Literal[0] | Literal[1] | Literal[2]
    ) -> tuple[int, int, int | float | complex]:

        readnum = file.readline()
        readnum = readnum.split()

        if vartype == 0:
            return (int(readnum[0]), int(readnum[1]), int(readnum[2]))

        if vartype == 1:
            return (int(readnum[0]), int(readnum[1]), float(readnum[2]))

        if vartype == 2:
            return (
                int(readnum[0]),
                int(readnum[1]),
                float(readnum[2]) + complex(float(readnum[3])),
            )

        raise AssertionError()

    @staticmethod
    def read_mtx(filename: str | Path) -> NDArray[DTYPE]:
        """Read matrix data from file."""

        with open(filename, "r") as file:
            line = str(file.readline())
            firstline = line.split()
            if firstline[1] == "matrix":
                shape = 0
            elif firstline[1] == "vector":
                shape = 1
            if firstline[2] == "array":
                descr = 0
            elif firstline[2] == "coordinate":
                descr = 1
            if firstline[3] == "integer":
                vartype = 0
            elif firstline[3] == "real":
                vartype = 1
            elif firstline[3] == "complex":
                vartype = 2
            if firstline[4] == "general":
                symtype = 0
            elif firstline[4] == "symmetric":
                symtype = 1
            elif firstline[4] == "skew-symmetric":
                symtype = 2
            elif firstline[4] == "hermitian":
                symtype = 3
            while line[0] == "%" or len(line) == 1:
                line = file.readline()
            line = list(map(int, line.split()))
            if shape == 0 and descr == 0:
                rows = line[0]
                cols = line[1]
                if vartype == 0:
                    retval = np.zeros((rows, cols), dtype=int)
                elif vartype == 1:
                    retval = np.zeros((rows, cols), dtype=float)
                elif vartype == 2:
                    retval = np.zeros((rows, cols), dtype=complex)
                if symtype == 0:
                    for i1 in range(cols):
                        for i2 in range(rows):
                            retval[i2][i1] = getnum(file, vartype)
                elif symtype == 1 and rows == cols:
                    for i1 in range(cols):
                        for i2 in range(i1, rows):
                            retval[i2][i1] = getnum(file, vartype)
                            retval[i1][i2] = retval[i2][i1]
                elif symtype == 2 and rows == cols:
                    for i1 in range(0, cols):
                        for i2 in range(i1 + 1, rows):
                            retval[i2][i1] = getnum(file, vartype)
                            retval[i1][i2] = -retval[i2][i1]
                elif symtype == 3 and rows == cols:
                    for i1 in range(0, cols):
                        for i2 in range(i1, rows):
                            retval[i2][i1] = getnum(file, vartype)
                            retval[i1][i2] = np.conj(retval[i2][i1])
            if shape == 0 and descr == 1:
                rows = line[0]
                cols = line[1]
                nonzeros = line[3]
                if vartype == 0:
                    retval = np.zeros((cols, rows), dtype=int)
                elif vartype == 1:
                    retval = np.zeros((cols, rows), dtype=float)
                elif vartype == 2:
                    retval = np.zeros((cols, rows), dtype=complex)
                for i1 in range(nonzeros):
                    entry = MtxIO.get_3_numbers(file, vartype)
                    retval[entry[1]][entry[0]] = entry[2]
                if symtype == 1 and rows == cols:
                    for i1 in range(rows):
                        for i2 in range(i1, cols):
                            if np.abs(retval[i1][i2]) != 0:
                                retval[i2][i1] = retval[i1][i2]
                            else:
                                retval[i1][i2] = retval[i2][i1]
                elif symtype == 2 and rows == cols:
                    for i1 in range(0, cols):
                        for i2 in range(i1 + 1, rows):
                            if np.abs(retval[i1][i2]) != 0:
                                retval[i2][i1] = -retval[i1][i2]
                            else:
                                retval[i1][i2] = -retval[i2][i1]
                elif symtype == 3 and rows == cols:
                    for i1 in range(0, cols):
                        for i2 in range(i1, rows):
                            if np.abs(retval[i1][i2]) != 0:
                                retval[i2][i1] = np.conj(retval[i1][i2])
                            else:
                                retval[i1][i2] = np.conj(retval[i2][i1])

        return retval

    @staticmethod
    def write_mtx(filename, lista, vartype):
        if vartype == 1:
            vartype1 = "real"
        elif vartype == 2:
            vartype1 = "complex"
        elif vartype == 0:
            vartype1 = "integer"
        with open(filename, "w") as file:
            file.write(" ".join(["%%MatrixMarket matrix array", vartype1, "general\n"]))
            file.write("%Generated by CSSFinder\n")
            file.write(str(len(lista)))
            file.write("  ")
            file.write(str(len(lista[0])))
            file.write("\n")
            for i1 in range(len(lista[0])):
                for i2 in range(len(lista)):
                    if vartype == 2:
                        file.write(str(np.real(lista[i2][i1])))
                        file.write("   ")
                        file.write(str(np.imag(lista[i2][i1])))
                        file.write("\n")
                    else:
                        file.write(str(lista[i2][i1]))
                        file.write("\n")
            file.close()


# Random states:
# n qubit state
def Random2FS(n):
    q1 = MtxFn.normalize(MtxFn.generate(2))
    if n > 1:
        for l1 in range(n - 1):
            q1 = MtxFn.outer(q1, MtxFn.normalize(MtxFn.generate(2)))
    return MtxFn.project(q1)


# n quDit state
def RandomdFS(d, n):
    q1 = MtxFn.normalize(MtxFn.generate(d))
    for l1 in range(n - 1):
        q1 = Outer(q1, MtxFn.normalize(MtxFn.generate(d)))
    return MtxFn.project(q1)


# biseparable state
def RandomBS(d1, d2):
    return MtxFn.project(
        MtxFn.outer(
            MtxFn.normalize(MtxFn.generate(d1)), MtxFn.normalize(MtxFn.generate(d2))
        )
    )


# biseparable state with three quDits
def Random3P(d1, swaps, i):
    if i == 0:
        # aBC
        return RandomBS(d1, d1 * d1)
    if i == 1:
        # AbC
        return MtxFn.rotate(RandomBS(d1, d1 * d1), swaps[0])
    if i == 2:
        # ABc
        return RandomBS(d1 * d1, d1)


# biseparable for  4 quDits
def Random4P(d1, swaps, i):
    if i == 0:
        # aBCD
        return RandomBS(d1, d1 * d1 * d1)
    if i == 1:
        # AbCD
        return MtxFn.rotate(RandomBS(d1, d1 * d1 * d1), swaps[0])
    if i == 2:
        # ABcD
        return MtxFn.rotate(RandomBS(d1 * d1 * d1, d1), swaps[3])
    if i == 3:
        # ABCd
        return RandomBS(d1 * d1 * d1, d1)
    if i == 4:
        # abCD
        return RandomBS(d1 * d1, d1 * d1)
    if i == 5:
        # aBcD
        return MtxFn.rotate(RandomBS(d1 * d1, d1 * d1), swaps[2])
    if i == 6:
        # aBCd
        return MtxFn.rotate(RandomBS(d1 * d1, d1 * d1), swaps[1])


# Rendom Unitaries
# Biseparability
def RandomUBS(a, d1, d2):
    if a == 0:
        rubsp1 = (
            math.cos(0.01 * math.pi) + complex(0, 1) * math.sin(0.01 * math.pi) - 1
        ) * MtxFn.project(MtxFn.normalize(MtxFn.generate(d1))) + MtxFn.ident_mtx(d1)
        return MtxFn.kronecker(rubsp1, MtxFn.ident_mtx(int(d2)))
    if a == 1:
        rubsp1 = (
            math.cos(0.01 * math.pi) + complex(0, 1) * math.sin(0.01 * math.pi) - 1
        ) * MtxFn.project(MtxFn.normalize(MtxFn.generate(d2))) + MtxFn.ident_mtx(d2)
        return MtxFn.kronecker(MtxFn.ident_mtx(int(d1)), rubsp1)
        # return(Kronecker(IdMatrix(d1),unitatry_group.rvs(d2)))


# n qubits
def RandomU2FS(n, j):
    # p1=unitary_group.rvs(2)
    p1 = (
        math.cos(0.01 * math.pi) + complex(0, 1) * math.sin(0.01 * math.pi) - 1
    ) * Random2FS(1) + MtxFn.ident_mtx(2)
    return MtxFn.expand2FS(p1, n, j)


# n quDits
def RandomUdFS(d, n, j):
    # p1=unitary_group.rvs(d)
    p1 = (
        math.cos(0.01 * math.pi) + complex(0, 1) * math.sin(0.01 * math.pi) - 1
    ) * RandomdFS(d, 1) + MtxFn.ident_mtx(d)
    return MtxFn.expand_d_FS(p1, d, n, j)


# Optimizers
# biseparability
def OptimizeBS(rho2, rho3, d1, d2):
    pp1 = MtxFn.product(rho2, rho3)
    for obsj1 in range(5 * d1 * d2):
        U = RandomUBS(obsj1 % 2, d1, d2)
        rho2a = MtxFn.rotate(rho2, U)
        if pp1 > MtxFn.product(rho2a, rho3):
            U = U.conj().T
            rho2a = MtxFn.rotate(rho2, U)
        while MtxFn.product(rho2a, rho3) > pp1:
            rho2b = rho2a
            pp1 = MtxFn.product(rho2b, rho3)
            rho2a = MtxFn.rotate(rho2a, U)
    return rho2a


# 3-partite entanglement
def Optimized3P(rho2, swaps, rho3, d1, i1):
    if i1 == 0:
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**6):
            U = RandomUBS(j1 % 2, d1, d1 * d1)
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    if i1 == 1:
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**6):
            U = MtxFn.rotate(RandomUBS(j1 % 2, d1, d1 * d1), swaps[0])
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    if i1 == 2:
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**6):
            U = RandomUBS(j1 % 2, d1 * d1, d1)
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    return rho2a


# four partite entanglement
# swap12,swap13,swap23,swap34
def Optimized4P(rho2, rho3, swaps, d1, i1):
    if i1 == 0:
        # aBCD
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**8):
            U = RandomUBS(j1 % 2, d1, d1**3)
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    if i1 == 1:
        # AbCD
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**8):
            U = MtxFn.rotate(RandomUBS(j1 % 2, d1, d1 * d1 * d1), swaps[0])
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    if i1 == 2:
        # ABcD
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**8):
            U = MtxFn.rotate(RandomUBS(j1 % 2, d1**3, d1), swaps[3])
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    if i1 == 3:
        # ABCd
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**8):
            U = RandomUBS(j1 % 2, d1**3, d1)
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    if i1 == 4:
        # abCD
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**8):
            U = RandomUBS(j1 % 2, d1 * d1, d1 * d1)
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    if i1 == 5:
        # aBcD
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**8):
            U = MtxFn.rotate(RandomUBS(j1 % 2, d1 * d1, d1 * d1), swaps[2])
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    if i1 == 6:
        # aBCd
        pp1 = MtxFn.product(rho2, rho3)
        for j1 in range(5 * d1**8):
            U = MtxFn.rotate(RandomUBS(j1 % 2, d1 * d1, d1 * d1), swaps[1])
            rho2a = MtxFn.rotate(rho2, U)
            if pp1 > MtxFn.product(rho2a, rho3):
                U = U.conj().T
                rho2a = MtxFn.rotate(rho2, U)
            while MtxFn.product(rho2a, rho3) > pp1:
                rho2b = rho2a
                pp1 = MtxFn.product(rho2b, rho3)
                rho2a = MtxFn.rotate(rho2a, U)
    return rho2a


# n qubits
def Optimize2FS(rho2, rho3, n):
    pp1 = MtxFn.product(rho2, rho3)
    for j1 in range(100 * n):
        U = RandomU2FS(n, j1 % n)
        rho2a = MtxFn.rotate(rho2, U)
        if pp1 > MtxFn.product(rho2a, rho3):
            U = U.conj().T
            rho2a = MtxFn.rotate(rho2, U)
        while MtxFn.product(rho2a, rho3) > pp1:
            rho2b = rho2a
            pp1 = MtxFn.product(rho2b, rho3)
            rho2a = MtxFn.rotate(rho2a, U)
    return rho2a


# n quDits
def OptimizedFS(rho2, rho3, ddd1, n):
    pp1 = MtxFn.product(rho2, rho3)
    for j1 in range(20 * ddd1 * ddd1 * n):
        U = RandomUdFS(ddd1, n, j1 % n)
        rho2a = MtxFn.rotate(rho2, U)
        if pp1 > MtxFn.product(rho2a, rho3):
            U = U.conj().T
            rho2a = MtxFn.rotate(rho2, U)
        while MtxFn.product(rho2a, rho3) > pp1:
            rho2b = rho2a
            pp1 = MtxFn.product(rho2b, rho3)
            rho2a = MtxFn.rotate(rho2a, U)
    return rho2a


# SWAP GATES at dimension d
def swap123(d):
    temp = np.zeros((d**3, d**3), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                temp[i1 * d**2 + i2 * d + i3][i2 * d**2 + i1 * d + i3] = 1
    return temp


# def swap133(d):
#     temp=np.zeros((d**3,d**3),dtype=complex)
#     for i1 in range(d):
#         for  i2 in range(d):
#             for i3 in range(d):
#                 temp[i1*d**2+i2*d+i3][i3*d**2+i2*d+i1]=1
#     return(temp)

# def swap233(d):
#     temp=np.zeros((d**3,d**3),dtype=complex)
#     for i1 in range(d):
#         for  i2 in range(d):
#             for i3 in range(d):
#                 temp[i1*d**2+i2*d+i3][i1*d**2+i3*d+i2]=1
#     return(temp)


def swap124(d):
    temp = np.zeros((d**4, d**4), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                for i4 in range(d):
                    temp[i1 * d**3 + i2 * d**2 + i3 * d + i4][
                        i2 * d**3 + i1 * d**2 + i3 * d + i4
                    ] = 1
    return temp


def swap134(d):
    temp = np.zeros((d**4, d**4), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                for i4 in range(d):
                    temp[i1 * d**3 + i2 * d**2 + i3 * d + i4][
                        i3 * d**3 + i2 * d**2 + i1 * d + i4
                    ] = 1
    return temp


# def swap144(d):
#     temp=np.zeros((d**4,d**4),dtype=complex)
#     for i1 in range(d):
#         for  i2 in range(d):
#             for i3 in range(d):
#                 for i4 in range(d):
#                     temp[i1*d**3+i2*d**2+i3*d+i4][i4*d**3+i2*d**2+i3*d+i1]=1
#     return(temp)


def swap234(d):
    temp = np.zeros((d**4, d**4), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                for i4 in range(d):
                    temp[i1 * d**3 + i2 * d**2 + i3 * d + i4][
                        i1 * d**3 + i3 * d**2 + i2 * d + i4
                    ] = 1
    return temp


# def swap244(d):
#     temp=np.zeros((d**4,d**4),dtype=complex)
#     for i1 in range(d):
#         for  i2 in range(d):
#             for i3 in range(d):
#                 for i4 in range(d):
#                     temp[i1*d**3+i2*d**2+i3*d+i4][i1*d**3+i4*d**2+i3*d+i4]=1
#     return(temp)


def swap344(d):
    temp = np.zeros((d**4, d**4), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                for i4 in range(d):
                    temp[i1 * d**3 + i2 * d**2 + i3 * d + i4][
                        i1 * d**3 + i2 * d**2 + i4 * d + i3
                    ] = 1
    return temp


# Initialize files
# read rho
def Initrho0(prefix):
    #    return(scipy.io.mmread("_".join([prefix,"in.mtx"])))
    return MtxIO.read_mtx("_".join([prefix, "in.mtx"]))


# read or generate rho1
def Initrho1(prefix, rho, mode, d1, vis):
    if exists("".join([prefix, "_out_", str(vis), ".mtx"])):
        #        return(scipy.io.mmread("".join([prefix,"_out_",str(vis),".mtx"])))
        return MtxIO.read_mtx("".join([prefix, "_out_", str(vis), ".mtx"]))
    elif exists(
        "".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])
    ):
        return MtxIO.read_mtx(
            "".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])
        )
    else:
        rhoa = np.zeros((len(rho), len(rho)), dtype=complex)
        for j1 in range(len(rho)):
            rhoa[j1][j1] = rho[j1][j1]
        return rhoa


# Too slow convergence error
def tooslow(counter, prefix, vis):
    with open(
        "".join([prefix, "_abort_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]),
        "w",
    ) as file:
        file.write("The program has done 10% of planned trails, but it found only ")
        file.write(str(counter))
        file.write(" corrections and it was terminated.\n")
        file.write("Increase the number of trails or decrease the visibility.\n")
        file.close()


def showtooslow():
    if True:
        print(
            " ███▄    █ ▓█████ ██▒   █▓▓█████  ██▀███       ▄████  ██▓ ██▒   █▓▓█████     █    ██  ██▓███  "
        )
        if True:
            print(
                " ██ ▀█   █ ▓█   ▀▓██░   █▒▓█   ▀ ▓██ ▒ ██▒    ██▒ ▀█▒▓██▒▓██░   █▒▓█   ▀     ██  ▓██▒▓██░  ██▒"
            )
            print(
                "▓██  ▀█ ██▒▒███   ▓██  █▒░▒███   ▓██ ░▄█ ▒   ▒██░▄▄▄░▒██▒ ▓██  █▒░▒███      ▓██  ▒██░▓██░ ██▓▒"
            )
        print(
            "▓██▒  ▐▌██▒▒▓█  ▄  ▒██ █░░▒▓█  ▄ ▒██▀▀█▄     ░▓█  ██▓░██░  ▒██ █░░▒▓█  ▄    ▓▓█  ░██░▒██▄█▓▒ ▒"
        )
        print(
            "▒██░   ▓██░░▒████▒  ▒▀█░  ░▒████▒░██▓ ▒██▒   ░▒▓███▀▒░██░   ▒▀█░  ░▒████▒   ▒▒█████▓ ▒██▒ ░  ░"
        )
        if True:
            print(
                "░ ▒░   ▒ ▒ ░░ ▒░ ░  ░ ▐░  ░░ ▒░ ░░ ▒▓ ░▒▓░    ░▒   ▒ ░▓     ░ ▐░  ░░ ▒░ ░   ░▒▓▒ ▒ ▒ ▒▓▒░ ░  ░"
            )
            print(
                "░ ░░   ░ ▒░ ░ ░  ░  ░ ░░   ░ ░  ░  ░▒ ░ ▒░     ░   ░  ▒ ░   ░ ░░   ░ ░  ░   ░░▒░ ░ ░ ░▒ ░"
            )
        print(
            "   ░   ░ ░    ░       ░░     ░     ░░   ░    ░ ░   ░  ▒ ░     ░░     ░       ░░░ ░ ░ ░░"
        )
        print(
            "         ░    ░  ░     ░     ░  ░   ░              ░  ░        ░     ░  ░      ░"
        )
    print("                       ░                                       ░")


# Wrong dimension
def wrongdim(prefix, mode):
    with open("".join([prefix, "_wrong_dim_.txt"]), "w") as file:
        file.write("Dimension of the imput state not compatible with the declared mode")
        file.close()


# Read symmetry transformations
def DefineSym(prefix):
    symlist1 = []
    symlist2 = []
    symj1 = 0
    symj2 = 0
    if exists("_".join([prefix, "sym_0_0.mtx"])):
        symflag = True
    while exists("_".join([prefix, "sym", str(symj1), "0.mtx"])):
        while exists("".join([prefix, "_sym_", str(symj1), "_", str(symj2), ".mtx"])):
            #            symlist2.append(scipy.io.mmread("".join([prefix,"_sym_",str(symj1),"_",str(symj2),".mtx"])))
            symlist2.append(
                MtxIO.read_mtx(
                    "".join([prefix, "_sym_", str(symj1), "_", str(symj2), ".mtx"])
                )
            )
            symj2 = symj2 + 1
        symj2 = 0
        symlist1.append(symlist2)
        symj1 = symj1 + 1
    return symlist1


# Apply symmetries to a state
def ApplySym(rho, symlist1):
    asrho0 = rho
    for asj1 in range(len(symlist1)):
        for asj2 in range(len(symlist1[asj1])):
            asrho0 = asrho0 + MtxFn.rotate(asrho0, symlist1[asj1][asj2])
        asrho0 = asrho0 / np.trace(asrho0)
    return asrho0


# Define projection
def DefineProj(projflag, prefix):
    if exists("_".join([prefix, "proj", ".mtx"])):
        projflag = True
        #        return(scipy.io.mmread("_".join([prefix,"proj.mtx"])))
        return MtxIO.read_mtx("_".join([prefix, "proj.mtx"]))
    else:
        return [[]]


# Gilbert algorithm
def Gilbert(mode, prefix, vis, rho, steps, corrs, d1, d2, verboseflag):
    rho1 = Initrho1(prefix, rho, mode, d1, vis)
    symflag = False
    symlist = DefineSym(prefix)
    if len(symlist) > 0:
        symflag = True
        print(len(symlist))
    projflag = False
    proj1 = DefineProj(projflag, prefix)
    # if symflag==True:
    #     rho1=ApplySym(rho1, symlist)
    if projflag == True:
        rho1 = MtxFn.rotate(rho1, proj1)
    lastcorr = 0
    currentcorr = 0
    ll = []
    counter = 0
    flag = 0
    trail = 0
    if exists("".join([prefix, "_list_", str(vis), ".mtx"])) and exists(
        "".join([prefix, "_out_", str(vis), ".mtx"])
    ):
        #        ll2=scipy.io.mmread("".join([prefix,"_list_",str(vis),".mtx"]))
        ll2 = MtxIO.read_mtx("".join([prefix, "_list_", str(vis), ".mtx"]))
        trail = int(ll2[len(ll2) - 1][0])
        counter = int(ll2[len(ll2) - 1][1])
        for i3 in range(len(ll2)):
            ll.append([int(ll2[i3][0]), int(ll2[i3][1]), ll2[i3][2]])
        if len(ll) == 1:
            currentcorr = ll[0][0]
        elif len(ll) > 1:
            currentcorr = ll[len(ll) - 1][0]
            lastcorr = ll[len(ll) - 2][0]
    if exists(
        "".join([prefix, "_list_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])
    ) and exists(
        "".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])
    ):
        #        ll2=scipy.io.mmread("".join([prefix,"_list_",str(vis),".mtx"]))
        ll2 = MtxIO.read_mtx(
            "".join([prefix, "_list_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])
        )
        trail = int(ll2[len(ll2) - 1][0])
        counter = int(ll2[len(ll2) - 1][1])
        for i3 in range(len(ll2)):
            ll.append([int(ll2[i3][0]), int(ll2[i3][1]), ll2[i3][2]])
        if len(ll) == 1:
            currentcorr = ll[0][0]
        elif len(ll) > 1:
            currentcorr = ll[len(ll) - 1][0]
            lastcorr = ll[len(ll) - 2][0]
    if steps < 0:
        steps = trail - steps
        steps = steps - steps % 10
    if corrs < 0:
        corrs = counter - corrs
    corrs = corrs - corrs % 50
    now = datetime.now()
    if verboseflag == True:
        print(
            now.strftime("%d/%m/%Y %H:%M:%S"),
            " Report: proceeding with",
            abs(steps) - trail,
            " iterations and ",
            abs(corrs) - counter,
            " corrections.",
        )
    realflag = True
    for realcheck1 in range(len(rho)):
        for realcheck2 in range(len(rho)):
            if np.imag(rho[realcheck1][realcheck2]) != 0:
                realflag = False
                break
    if realflag == True and verboseflag == True:
        print("\n")
        print(
            "Input state strictly real. Imaginary parts of the output state will be discarded"
        )
    aa1 = MtxFn.product(rho, rho)
    aa4 = 2 * MtxFn.product(rho, rho1)
    aa6 = MtxFn.product(rho1, rho1)
    rho3 = rho - rho1
    dd1 = MtxFn.product(rho1, rho3)
    if mode == 3:
        swaps = [swap123(d1)]
    elif mode == 4:
        swaps = [swap124(d1), swap134(d1), swap234(d1), swap344(d1)]
    # else:
    #     swaps=[]
    if len(ll) == 0 or ll[len(ll) - 1][2] > 0.000001:
        carryonflag = True
    else:
        carryonflag = False
    while (
        trail <= abs(steps)
        and counter <= abs(corrs)
        and (len(ll) == 0 or ll[len(ll) - 1][2] > 0.0000001)
    ):
        trail = trail + 1
        if currentcorr - lastcorr > (steps - trail):
            if verboseflag == True:
                print("Too few steps left. Quitting")
            break
        if divmod(10 * trail / abs(steps), 1)[1] == 0:
            if counter < 50:
                tooslow(counter, prefix, vis)
                if verboseflag == True:
                    showtooslow()
                break
            now = datetime.now()
            if verboseflag == True:
                print(
                    now.strftime("%d/%m/%Y %H:%M:%S"),
                    " Report: done",
                    trail,
                    "/",
                    abs(steps),
                )
        # if mode==0:
        #     rho2=Random2FS(d1)
        if mode == 2:
            rho2 = RandomBS(d1, d2)
        if mode == 1:
            rho2 = RandomdFS(d1, d2)
        if mode == 3:
            rho2 = Random3P(d1, swaps, trail % 3)
        if mode == 4:
            rho2 = Random4P(d1, swaps, trail % 7)
        if not (mode in range(1, 5)):
            if verboseflag == True:
                print("Mode ", mode, "does not exist!")
            DisplayHelp()
            break
        if MtxFn.product(rho2, rho3) > dd1:
            # if mode==0:
            #     rho2=Optimize2FS(rho2, rho3, d1)
            if mode == 2:
                rho2 = OptimizeBS(rho2, rho3, d1, d2)
            if mode == 1:
                rho2 = OptimizedFS(rho2, rho3, d1, d2)
            if mode == 3:
                rho2 = Optimized3P(rho2, swaps, rho3, d1, trail % 3)
            if mode == 4:
                rho2 = Optimized4P(rho2, rho3, swaps, d1, trail % 7)
            if realflag == True:
                rho2 = np.real(rho2)
            if symflag == True:
                rho1 = ApplySym(rho1, symlist)
            if projflag == True:
                rho1 = MtxFn.rotate(rho1, proj1)
                rho1 = rho1 / np.trace(rho1)
            aa3 = MtxFn.product(rho2, rho2)
            aa2 = 2 * MtxFn.product(rho, rho2)
            aa5 = 2 * MtxFn.product(rho1, rho2)
            #            bb1=aa1-aa2+aa3
            bb2 = -aa4 + aa2 + aa5 - 2 * aa3
            bb3 = aa6 - aa5 + aa3
            cc1 = -bb2 / (2 * bb3)
            #            cc2=-bb2*bb2/(4*bb3)+bb1
            if 0 <= cc1 and cc1 <= 1:
                rho1 = cc1 * rho1 + (1 - cc1) * rho2
                if symflag == True and counter % 50 == 0:
                    rho1 = ApplySym(rho1, symlist)
                counter = counter + 1
                if counter % 2 == 0 and verboseflag == True:
                    marker = counter % 50
                    print(marker, end=" ")
                    if marker == 0:
                        print("")
                    # if marker in [2,4,6,8]:
                    #     # print("\b",end="")
                    #     # print("\b",end="")
                    #     # print("",end="")
                    #     print(marker,end=" ")
                    # elif marker==0:
                    #     print("")
                    # #     print("\b",end="")
                    # #     print("\b",end="")
                    # else:
                    #     # print("\b",end="")
                    #     # print("\b",end="")
                    #     print(marker,end=" ")
                rho3 = rho - rho1
                aa4 = 2 * MtxFn.product(rho, rho1)
                aa6 = MtxFn.product(rho1, rho1)
                dd1 = aa4 / 2 - aa6
                flag = 1
            if 0 > cc1 or cc1 > 1:
                flag = 0
            if counter % 50 == 0 and flag == 1:
                if verboseflag == True:
                    now = datetime.now()
                    print(
                        now.strftime("%H:%M:%S: "),
                        "Trails:",
                        trail,
                        " Corrections:",
                        counter,
                        "D^2:",
                        MtxFn.product(rho3, rho3),
                    )
                ll.append([trail, counter, MtxFn.product(rho3, rho3)])
                #                scipy.io.mmwrite("".join([prefix,"_out_",str(vis),".mtx"]),rho1)
                MtxIO.write_mtx(
                    "".join(
                        [
                            prefix,
                            "_out_",
                            str(mode),
                            "_",
                            str(d1),
                            "_",
                            str(vis),
                            ".mtx",
                        ]
                    ),
                    rho1,
                    2,
                )
                #                scipy.io.mmwrite("".join([prefix,"_list_",str(vis),".mtx"]),ll)
                MtxIO.write_mtx(
                    "".join(
                        [
                            prefix,
                            "_list_",
                            str(mode),
                            "_",
                            str(d1),
                            "_",
                            str(vis),
                            ".mtx",
                        ]
                    ),
                    ll,
                    1,
                )
                lastcorr = currentcorr
                currentcorr = trail
    return ll


def invert(c, a):
    return 1 / (c - a)


def mean(l):
    return sum(l) / len(l)


def R(l, a):
    ll1 = list(map(lambda x1: invert(x1, a), l))
    return (
        mean(list(map(lambda x1, x2: x1 * x2, ll1, list(range(len(l))))))
        - mean(ll1) * mean(list(range(len(l))))
    ) / math.sqrt(
        (mean(list(map(lambda x: x**2, ll1))) - mean(ll1) ** 2)
        * (
            mean(list(map(lambda x: x**2, list(range(len(l))))))
            - mean(list(range(len(l)))) ** 2
        )
    )


def listshift(l1, a1):
    return list(map(lambda x: x - a1, l1))


def cov(l1, l2):
    return mean(
        list(
            map(
                lambda x1, x2: x1 * x2, listshift(l1, mean(l1)), listshift(l2, mean(l2))
            )
        )
    )


def trend(l1, l2):
    l1a = list(map(lambda x: math.log(x), l1))
    l2a = list(map(lambda x: math.log(x), l2))
    return cov(l1a, l2a) / cov(l1a, l1a)


def offset(l1, l2):
    l1a = list(map(lambda x: math.log(x), l1))
    l2a = list(map(lambda x: math.log(x), l2))
    return mean(l2a) - mean(l1a) * trend(l1, l2)


def findmaximum(ll):
    list1 = list(map(lambda x: x[2], ll))
    list2 = []
    for j1 in range(int(len(list1) / 2), len(list1)):
        list2.append(list1[j1])
    aaa1 = list2[len(list2) - 1] - 0.000001
    step1 = aaa1 / 10000
    while R(list2, aaa1 - step1) > R(list2, aaa1) and aaa1 > 0:
        aaa1 = aaa1 - step1
    return aaa1


def makeshortreport(prefix, ll, mode, d1, vis):
    ll10 = list(map(lambda x: x[0], ll))
    ll11 = list(map(lambda x: x[1], ll))
    kk = findmaximum(ll)
    ll12 = []
    for j1 in range(int(2 * len(ll) / 3), len(ll)):
        ll12.append(ll[j1][2])
    with open(
        "".join([prefix, "_report_", str(mode), "_", str(d1), "_", str(vis), ".txt"]),
        "w",
    ) as file:
        file.write("Basing on decay, the squared HS distance is estimsated to be ")
        file.write(str(kk))
        file.write(" (R=")
        file.write(str(R(ll12, kk)))
        file.write(")\n")
        file.write("The dependence between corrs and trail is approximately:\n")
        file.write("corr=trail^")
        file.write(str(trend(ll10, ll11)))
        file.write("*")
        file.write(str(math.exp(offset(ll10, ll11))))
        file.write("\n-----------------\n")
        file.close()


def makelongreport(prefix, mode, vis, swaps, d1, d2, ll, verboseflag):
    # optw=OptimizeW(prefix, mode, vis, swaps, d1, d2,verboseflag)
    # wdist0=WitnessDist(prefix, vis, optw,verboseflag)
    # with open("".join([prefix,"_report_",str(vis),".txt"]),'a') as file:
    #     if wdist0==-2:
    #         file.write("The algorithm did not yield a valid entanglement witness.")
    #     else:
    #         file.write("The squared distance based on entanglement witness is ")
    #         file.write(str(wdist0))
    #     file.close()
    rhoa = MtxIO.read_mtx("".join([prefix, "_in.mtx"]))
    #    rhob=scipy.io.mmread("".join([prefix,"_out_",str(vis),".mtx"]))
    rhob = MtxIO.read_mtx(
        "".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])
    )
    rhoa = vis * rhoa + (1 - vis) * MtxFn.ident_mtx(len(rhoa)) / len(rhoa)
    witness = rhoa - rhob
    MtxIO.write_mtx(
        "".join([prefix, "_witness_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]),
        witness,
        2,
    )


def OptimizeW(prefix, mode, vis, swaps, d1, d2, verboseflag):
    l = -1
    #    rhoa=scipy.io.mmread("_".join([prefix,"in.mtx"]))
    rhoa = MtxIO.read_mtx("_".join([prefix, "in.mtx"]))
    rhob = Initrho1(prefix, rhoa, vis)
    witness = rhoa - rhob
    #     zasieg=1000*d1
    if mode == 2:
        zasieg = 1500
    if mode == 1:
        zasieg = 1500
    if mode == 3:
        zasieg = 1500
    if mode == 4:
        zasieg = 1500
    if verboseflag == True:
        print("Optimizing the potential witness operator. Number of trails:", zasieg)
    for owi1 in range(zasieg):
        # if mode==0:
        #     w1=Random2FS(d1)
        #     l.append(Optimize2FS(w1,witness, d1))
        if owi1 % 100 == 0 and verboseflag:
            print("trail:", owi1)
        if mode == 2:
            w1 = RandomBS(d1, d2)
            rho2a = OptimizeBS(w1, witness, d1, d2)
            l1 = MtxFn.product(rho2a, witness)
            if l1 > l:
                l = l1
        if mode == 1:
            w1 = RandomdFS(d1, d2)
            rho2a = OptimizedFS(w1, witness, d1, d2)
            l1 = MtxFn.product(rho2a, witness)
            if l1 > l:
                l = l1
        if mode == 3:
            for owi2 in range(3):
                w1 = Random3P(d1, swaps[0], owi2)
                rho2a = Optimize2FS(w1, witness, d1)
                l1 = MtxFn.product(rho2a, witness)
                if l1 > l:
                    l = l1
        if mode == 4:
            for owi2 in range(7):
                w1 = Random4P(d1, swaps[0], swaps[1], swaps[2], swaps[3], owi2)
                rho2a = Optimize2FS(w1, witness, d1)
                l1 = MtxFn.product(rho2a, witness)
                if l1 > l:
                    l = l1
    return l


def WitnessDist(prefix, vis, sepmax, verboseflag):
    #    rhoa=scipy.io.mmread("".join([prefix,"_in.mtx"]))
    rhoa = MtxIO.read_mtx("".join([prefix, "_in.mtx"]))
    #    rhob=scipy.io.mmread("".join([prefix,"_out_",str(vis),".mtx"]))
    rhob = MtxIO.read_mtx(
        "".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])
    )
    rhoa = vis * rhoa + (1 - vis) * MtxFn.ident_mtx(len(rhoa)) / len(rhoa)
    witness = rhoa - rhob
    wdist = (MtxFn.product(witness, rhoa) - sepmax) / math.sqrt(
        MtxFn.product(witness, witness)
    )
    if wdist < 0:
        if verboseflag == True:
            print("No entanglement witness found.")
    else:
        if verboseflag == True:
            print(
                "Witness-based estimated squared distance:", wdist**2, " (VERIFY!!!)"
            )
        #        scipy.io.mmwrite("".join([prefix,"_witness_",str(vis),".mtx"]),witness,"".join(["Estimated sqared distance:",str(wdist**2)]))
        MtxIO.write_mtx(
            "".join(
                [prefix, "_witness_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]
            ),
            witness,
            2,
        )
    return wdist**2


def DetectDim0(mode, totaldim, verboseflag):
    primes = PRIMES
    ddflag = False
    if mode == 1:
        for pr in primes:
            if math.log(totaldim, pr) == int(math.log(totaldim, pr)):
                if verboseflag == True:
                    print(
                        "Determined size: ",
                        pr,
                        " number of subsystems:",
                        int(math.log(totaldim, pr)),
                    )
                return [pr, int(math.log(totaldim, pr))]
                ddflag = True
                break
    elif mode == 2:
        if math.sqrt(totaldim) == int(math.sqrt(totaldim)):
            pr = int(math.sqrt(totaldim))
            if verboseflag == True:
                print("Determined sizes of subsytems: ", pr, int(totaldim / pr))
            return [int(math.sqrt(totaldim)), int(math.sqrt(totaldim))]
            ddflag = True
        else:
            for pr in primes:
                if totaldim % pr == 0:
                    if verboseflag == True:
                        print("Determined sizes of subsytems: ", pr, int(totaldim / pr))
                    return [pr, int(totaldim / pr)]
                    ddflag = True
                    break
    elif mode == 3:
        if totaldim ** (1.0 / 3) == int(totaldim ** (1.0 / 3)):
            if verboseflag == True:
                print(
                    "Determined size: ",
                    int(totaldim ** (1.0 / 3)),
                    " number of subsystems:",
                    3,
                )
            return [int(totaldim ** (1.0 / 3)), 3]
        ddflag = True
    elif mode == 4:
        if totaldim ** (1.0 / 4) == int(totaldim ** (1.0 / 4)):
            if verboseflag == True:
                print(
                    "Determined size: ",
                    int(totaldim ** (1.0 / 4)),
                    " number of subsystems:",
                    4,
                )
            return [int(totaldim ** (1.0 / 4)), 4]
        ddflag = True
    if ddflag == False:
        return [0, 0]


def DetectDim1(mode, totdim, d1, verboseflag):
    ddflag = False
    if mode == 1:
        if math.log(totdim, d1) == int(math.log(totdim, d1)):
            ddflag == True
            if verboseflag == True:
                print(
                    "Determined size: ",
                    d1,
                    " number of subsystems:",
                    int(math.log(totdim, d1)),
                )
            return int(math.log(totdim, d1))
    elif mode == 2:
        if totdim / d1 == int(totdim / d1):
            ddflag == True
            if verboseflag == True:
                print("Determined sizes", d1, ", ", int(totdim / d1))
            return int(totdim / d1)
    elif mode == 3:
        if d1**3 == totdim:
            ddflag == True
            return 3
    elif mode == 4:
        if d1**4 == totdim:
            ddflag == True
            return 4
    if ddflag == False:
        return 0


def DisplayHelp():
    print("CSSFinder mode verbose prefix vis steps corrs d1")
    print(
        "mode=0: full separability of an n-qubit state(d1 optional and can be arbitrary)"
    )
    print("mode=1: full separability of an n-quDit state")
    print("mode=2: separability of a bipirtite state")
    print("mode=3: genuine 3-partite entaglement of a 3-quDit state")
    print("mode=4: genuine 4-partite entaglement of a 3-quDit state")
    print("verbose=0: supress on-screen ouptut")
    print("verbose=1: show on-screen messages and reports")
    print("prefix: prefix of all used prefixs")
    print(
        "vis: visibility against white noise. Between 0 and 1. To be used when the algorithm is stuck"
    )
    print(
        "steps: number of attemps to correct the closest separable state. A negative number adds its value to pre-existing list of results"
    )
    print("steps are rounded down to a multiple of 10")
    print(
        "corrs: the maximal number of corrections to the closest separable state. A negative number adds its value to pre-existing list of results."
    )
    print("corrs are rounded down to a multiple of 50")
    print(
        "d1: the dimenssion of the first subsystem. Dimension of the other subsystem or the number of parties is deduced from the dimension of the input state."
    )
    print("Input:")
    print("prefix_in.mtx: the input state in MTX format")
    print(
        "prefix_sym_0_0.mtx,prefix_sym_0_1.mtx,...: symmetry unitaries applied to the output state in MTX format. The first number is the symmetry label, the second is the manifold. Optional"
    )
    print(
        "prefix_proj.mtx,prefix_sym_0_1.mtx,...: projectionsapplied to the output state in MTX format. Optional"
    )
    print("Output:")
    print(
        "prefix_mode_d1_out_vis.mtx: final sepa rable state (can be used as an initial separable state)"
    )
    print(
        "prefix_mode_d1_list_vis.mtx: number of steps, corrections, and the squared HS distance every 50 corrections"
    )
    print("If these files exist, the program will resume from the last record")
    print("prefix_report_mode_d1_vis.txt: The report file.")
    #    print("prefix_witness_mode_d1_vis.txt: entanglement witness candidate. Consult [Quantum Reports 2, 49].")
    print(
        "prefix_abort.txt: The error message if the algorithm was extremally slow (for some highly entangled states)."
    )


# Display the Logo
def DisplayLogo():
    print(" ██████╗███████╗███████╗███████╗██╗███╗   ██╗██████╗ ███████╗██████╗")
    if True:
        print("██╔════╝██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔══██╗██╔════╝██╔══██╗")
        if True:
            print(
                "██║     ███████╗███████╗█████╗  ██║██╔██╗ ██║██║  ██║█████╗  ██████╔╝"
            )
        print("██║     ╚════██║╚════██║██╔══╝  ██║██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗")
    print("╚██████╗███████║███████║██║     ██║██║ ╚████║██████╔╝███████╗██║  ██║")
    print(" ╚═════╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝")


def main(args):
    #     argnum=len(sys.argv)
    #     args=argstring
    argnum = len(args)
    verboseflag = True
    #     args=sys.argv
    argflag = False
    symflag = False
    if (argnum == 7 or argnum == 8) and (int(args[1]) in range(1, 5)):
        argflag = True
    if argflag == False:
        DisplayHelp()
    else:
        report = int(args[2])
        if report == 0:
            verboseflag = False
        if verboseflag == True:
            DisplayLogo()
        correctdimflag = True
        mode = int(args[1])
        prefix = args[3]
        vis = float(args[4])
        rho = Initrho0(prefix)
        totdim = len(rho)
        if argnum == 7:
            [d1, d2] = DetectDim0(mode, totdim, verboseflag)
        elif argnum == 8:
            d1 = int(args[7])
            d2 = DetectDim1(mode, totdim, d1, verboseflag)
        if d2 == 0:
            correctdimflag == False
            wrongdim(prefix, mode)
        else:
            rho = vis * rho + (1 - vis) * MtxFn.ident_mtx(totdim) / totdim
            rho1 = np.zeros((totdim, totdim))
            steps = int(args[5])
            if steps == 0:
                steps = -30000000
            corrs = int(args[6])
            if corrs == 0:
                if exists(
                    "".join(
                        [
                            prefix,
                            "_list_",
                            str(mode),
                            "_",
                            str(d1),
                            "_",
                            str(vis),
                            ".mtx",
                        ]
                    )
                ) and exists(
                    "".join(
                        [
                            prefix,
                            "_out_",
                            str(mode),
                            "_",
                            str(d1),
                            "_",
                            str(vis),
                            ".mtx",
                        ]
                    )
                ):
                    corrs = -500
                else:
                    coors = 2500
        if correctdimflag == True:
            ll1 = Gilbert(mode, prefix, vis, rho, steps, corrs, d1, d2, verboseflag)
            if len(ll1) < 10:
                if verboseflag == True:
                    print(
                        "The report can be generated only if more than 500 corrections were performed."
                    )
            else:
                makeshortreport(prefix, ll1, mode, d1, vis)
                if mode == 3:
                    swaps = [swap123(d1)]
                elif mode == 4:
                    swaps = [swap124(d1), swap134(d1), swap234(d1), swap344(d1)]
                else:
                    swaps = []
                makeshortreport(prefix, ll1, mode, d1, vis)
        #                    makelongreport(prefix, mode, vis, swaps, d1, d2, ll1,verboseflag)
        else:
            print("Input state dimensionality incompatiblie with declared mode.")
            DisplayHelp()


if __name__ == "__main__":
    main(sys.argv)
