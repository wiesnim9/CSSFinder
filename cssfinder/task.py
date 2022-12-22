from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Type, TypeVar


DEFAULT_TASK_OUT_DIR = Path.cwd() / "out"


class Mode(Enum):
    """List of possible execution modes."""

    FSNQ = "FSNQ"
    """Full separability of an n-quDit state."""

    FSNQ_PLUS = "FSNQ+"
    """Full separability of an n-quDit state (d1 optional and can be arbitrary)."""

    SBS = "SBS"
    """Separability of a bipartite state."""

    G3PE3Q = "G3PE3Q"
    """Genuine 3-partite entanglement of a 3-quDit state"""

    G4PE3Q = "G4PE3Q"
    """Genuine 4-partite entanglement of a 3-quDit state"""


_TaskT = TypeVar("_TaskT", bound="Task")


@dataclass(frozen=True)
class Task:
    """Task specification for algorithm."""

    mode: Mode
    visibility: float
    steps: int
    correlations: int
    input_dir: Path
    output_dir: Path

    @classmethod
    def new(  # pylint: disable=too-many-arguments
        cls: Type[_TaskT],
        mode: str | Mode,
        visibility: int | float | str,
        steps: int | float | str,
        correlations: int | float | str,
        input_dir: str | Path,
        output_dir: str | Path | None,
    ) -> _TaskT:
        """Create new Task instance with automatic field data validation.

        Parameters
        ----------
        mode : str | Mode
            Algorithm mode.
        visibility : int | float | str
            visibility against white noise.
        steps : int | float | str
            Number of attempts to correct the closest separable state.
        correlations : int | float | str
            Maximal number of corrections to the closest separable state.
        input_dir : str | Path
            Path to the directory containing input files.
        output_dir : str | Path | None
            Path to directory to put output files in.

        Returns
        -------
        _TaskT
            Task instance.
        """
        return cls(
            Mode(mode),
            min(1.0, max(0.0, float(visibility))),
            (int(steps) // 10) * 10,
            (int(correlations) // 50) * 50,
            Path(input_dir),
            Path(output_dir) if output_dir is not None else DEFAULT_TASK_OUT_DIR,
        )

    def describe(self) -> str:
        """Short task content description as string."""
        return f"<{self.mode.value}/{self.visibility}/{self.steps}/{self.correlations}/{self.input_dir}>"
