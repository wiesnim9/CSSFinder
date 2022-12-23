from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, TypeVar

from cssfinder.modes import DataType, Mode

DEFAULT_TASK_OUT_DIR = Path.cwd() / "out"


_TaskT = TypeVar("_TaskT", bound="Task")


@dataclass
class Task:
    """Task specification for algorithm."""

    mode: Mode
    visibility: float
    steps: int
    correlations: int

    input_dir: Path
    output_dir: Path

    size: Optional[int]
    sub_sys_size: Optional[int]
    data_type: DataType

    @classmethod
    def new(  # pylint: disable=too-many-arguments
        cls: Type[_TaskT],
        mode: str | Mode,
        visibility: int | float | str,
        steps: int | float | str,
        correlations: int | float | str,
        input_dir: str | Path,
        output_dir: str | Path | None,
        size: int | None,
        sub_sys_size: int | None,
        data_type: str | DataType,
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
        instance = cls(
            Mode(mode),
            min(1.0, max(0.0, float(visibility))),
            (int(steps) // 10) * 10,
            (int(correlations) // 50) * 50,
            Path(input_dir),
            Path(output_dir) if output_dir is not None else DEFAULT_TASK_OUT_DIR,
            size,
            sub_sys_size,
            DataType(data_type),  # type: ignore
        )
        return instance

    def describe(self) -> str:
        """Short task content description as string."""
        return (
            f"<{self.mode.value}/{self.visibility}/{self.steps}/"
            f"{self.correlations}/{self.input_dir}>"
        )

    def get_prefix(self) -> str:
        """File and directory prefix."""
        return self.input_dir.name

    def get_input_file(self) -> Path:
        """Path to file containing input state."""
        return self.input_dir / f"{self.get_prefix()}_in.mtx"
