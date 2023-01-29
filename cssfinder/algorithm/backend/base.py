"""This module contains base class for creating Gilbert algorithm backends (concrete
implementations) and exceptions which are expected to be raised from backends."""


from __future__ import annotations

from typing import Callable, Type, TypeVar

import numpy as np
import numpy.typing as npt

from cssfinder.io.v1_0_0.asset_loader import State
from cssfinder.project.v1_0_0.cssfproject import AlgoMode, Backend, Precision

BackendT = TypeVar("BackendT", bound="BackendBase")


class BackendBase:
    """Gilbert algorithm backend (implementation)."""

    backend_index: dict[tuple[Backend, Precision], Type[BackendBase]]

    def __init__(self, initial: State, mode: AlgoMode, visibility: float) -> None:
        self.initial = initial
        self.visibility = visibility
        self.mode = mode

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

    def run_epoch(self, iterations: int) -> None:
        """Run sequence of iterations without stopping to check any stop conditions."""
        raise NotImplementedError()

    @classmethod
    def select(cls, backend: Backend, precision: Precision) -> Type[BackendBase]:
        """Select one of the backends with fixed precision."""
        try:
            return cls.backend_index[(backend, precision)]
        except KeyError as exc:
            raise UnsupportedBackendError(
                f"Backend {backend.name!r} not supported."
            ) from exc

    @classmethod
    def register(
        cls, backend: Backend, precision: Precision
    ) -> Callable[[Type[BackendT]], Type[BackendT]]:
        """Register backend class as backend implementation."""

        def _register(backend_cls: Type[BackendT]) -> Type[BackendT]:
            cls.backend_index[(backend, precision)] = backend_cls

            return backend_cls

        return _register


class UnsupportedBackendError(Exception):
    """Raised for unsupported backend type."""
