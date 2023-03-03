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


"""CSSFinder uses its own project format allowing for file based customization of
parameters used by gilbert algorithm.

This file contains implementation of project configuration in 1.0.0 version.
"""


from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

import jsonref
from pydantic import ConstrainedStr, EmailStr, Field, validator

from cssfinder.base_model import CommonBaseModel
from cssfinder.enums import CaseInsensitiveEnum

if TYPE_CHECKING:
    from typing_extensions import Self


class CSSFProject(CommonBaseModel):
    """CSSFProject file specification."""

    meta: Meta
    """Project meta information like name and author."""

    tasks: dict[str, Task]
    """List of tasks within project which can be executed."""

    _file: Optional[Path] = None
    """Path to loaded project file."""

    _is_evaluated: bool = False

    def __init__(
        self, meta: Meta, tasks: list[Task] | dict[str, Task], *_: Any, **_k: Any
    ) -> None:
        super().__init__(meta=meta, tasks=tasks)

    @validator("tasks", pre=True, always=True)
    @classmethod
    def _validate_tasks(
        cls, value: Iterable[dict[str, Any]] | dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        if isinstance(value, dict):
            assert all(isinstance(v, dict) for v in value.values())
            return {str(k): dict(v) for k, v in value.items()}

        return {str(i): t for i, t in enumerate(value)}

    def eval_dynamic(self) -> None:
        """Evaluate dynamic path expressions."""
        if self._is_evaluated:
            return

        for task_name, task in self.tasks.items():
            task.eval_dynamic(self, task_name, task)

        self._is_evaluated = True

    def set_file_path(self, file: Path) -> None:
        """Set `cssfproject.json` file path."""
        self._file = file.expanduser().resolve()

    @property
    def file(self) -> Path:
        """Path to `cssfproject.json` file."""
        if self._file is None:
            return Path.cwd() / "cssfproject.json"
        return self._file

    @property
    def directory(self) -> Path:
        """Path to directory containing `cssfproject.json` file."""
        if self._file is None:
            return Path.cwd()
        return self._file.parent

    @property
    def output(self) -> Path:
        """Path to output directory for this project."""
        directory = self.directory / "output"
        directory.mkdir(0o764, True, True)
        return directory

    @classmethod
    def load_project(cls, file_or_directory: str | Path) -> Self:
        """Load CSSFinder project in at least 1.0.0 version.

        Parameters
        ----------
        file_or_directory : str | Path
            Either project file (cssfproject.json) or directory containing project file.

        Returns
        -------
        CSSFProject
            Project information container.

        Raises
        ------
        InvalidCSSFProjectContent
            Raised when project file content is not a dictionary.
        MalformedProjectFileError
            When content of project file is not valid json.
        """

        # Unify path type to Path
        file_or_directory = Path(file_or_directory).expanduser().resolve()

        # When points to directory, dir must contain cssfproject.json file
        if file_or_directory.is_dir():
            file_or_directory /= "cssfproject.json"
            file_path = file_or_directory
            del file_or_directory

        else:
            file_path = file_or_directory
            del file_or_directory

        logging.debug("Resolved project path to %r", file_path.as_posix())
        try:
            content = file_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ProjectFileNotFound(
                f"Make sure you path is correct: {file_path!r}"
            ) from exc

        try:
            content = jsonref.loads(content)
        except jsonref.JsonRefError as exc:
            raise MalformedProjectFileError() from exc

        if not isinstance(content, dict):
            logging.critical("Content of cssfproject.json file is not a dictionary.")
            raise InvalidCSSFProjectContent(content)

        project = cls(**content)
        project.set_file_path(file_path)
        project.eval_dynamic()

        return project

    def select_tasks(self, patterns: Optional[list[str]] = None) -> list[Task]:
        """Select all tasks matching list of patterns."""
        if patterns is None:
            return list(self.tasks.values())

        keys = set()

        for pattern in patterns:
            keys.update(fnmatch.filter(self.tasks.keys(), pattern))

        return [self.tasks[k] for k in keys]


class InvalidCSSFProjectContent(ValueError):
    """Raised by load_from() when file content is not a dictionary."""


class MalformedProjectFileError(ValueError):
    """Rased when project file content can't be correctly decoded."""


class ProjectFileNotFound(FileNotFoundError):
    """Raised when project file can't be found in expected place."""


class Meta(CommonBaseModel):
    """Project meta information."""

    author: str
    """Author full name."""

    email: EmailStr
    """Author email address."""

    name: str
    """Name of the project."""

    description: str
    """Description of the project."""

    version: SemVerStr
    """Version of the project."""


class SemVerStr(ConstrainedStr):
    """Semantic versioning string regex, see https://semver.org/"""

    regex = (
        r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-]"
        + r"[0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+("
        + r"[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )


class Task(CommonBaseModel):
    """This represents algorithm task."""

    gilbert: Optional[GilbertCfg] = Field(default=None)
    """Configuration of gilbert algorithm."""

    _output: Path = Field(default=Path.cwd())
    """Path default output directory."""

    _task_name: str = Field(default="")
    """Name of task assigned to it in project."""

    @property
    def output(self) -> Path:
        """Path to output directory of task."""
        return self._output

    @property
    def name(self) -> str:
        """Name of this task in project."""
        return self._task_name

    def eval_dynamic(self, project: CSSFProject, task_name: str, task: Task) -> None:
        """Evaluate dynamic path expressions."""
        if self.gilbert is not None:
            self.gilbert.eval_dynamic(project, task_name, task)
            self._output = project.output / task_name
            self._task_name = task_name


class GilbertCfg(CommonBaseModel):
    """Gilbert algorithm configuration container class."""

    mode: AlgoMode
    """Algorithm mode to use."""

    backend: Optional[BackendCfg] = Field(default=None)
    """Configuration of backend which will be used for execution.

    When backend configuration is not specified, numpy with double precision is used.
    """

    state: State | str | Path
    """Path to file containing initial state matrix."""

    runtime: RuntimeCfg
    """Configuration of runtime limits and parameters influencing algorithm run time."""

    resources: Optional[Resources] = Field(default=None)
    """Additional resources which may be used by algorithm."""

    @validator("resources", pre=True)
    @classmethod
    def _use_default_resources(cls, value: Optional[Resources]) -> Resources:
        if value is None:
            return Resources()
        return value

    @validator("backend", pre=True)
    @classmethod
    def _use_default_backend(cls, value: Optional[BackendCfg]) -> BackendCfg:
        if value is None:
            return BackendCfg(name=Backend.NumPy, precision=Precision.DOUBLE)
        return value

    def get_backend(self) -> BackendCfg:
        """Return resources object."""
        if self.backend is None:
            raise TypeError("Missing backend object.")
        return self.backend

    def get_resources(self) -> Resources:
        """Return resources object."""
        if self.resources is None:
            raise TypeError("Missing resources object.")
        return self.resources

    def eval_dynamic(self, project: CSSFProject, task_name: str, task: Task) -> None:
        """Evaluate dynamic path expressions."""

        if isinstance(self.state, str):
            self.state = State(file=self.state)

        elif isinstance(self.state, Path):
            self.state = State(file=self.state.expanduser().resolve().as_posix())

        self.state.eval_dynamic(project, task_name, task)
        self.get_resources().eval_dynamic(project, task_name, task)

    def get_state(self) -> State:
        """Return initial state information."""
        assert isinstance(self.state, State)
        return self.state


class AlgoMode(CaseInsensitiveEnum):
    """Mode of algorithm."""

    # pylint: disable=invalid-name

    FSnQd = "FSnQd"
    """Full separability of n-quDit state."""

    SBiPa = "SBiPi"
    """Separability of a bipartite state."""

    G3PaE3qD = "G3PaE3qD"
    """Genuine 3-partite entanglement of a 3-quDit state."""

    G4PaE3qD = "G4PaE3qD"
    """Genuine 4-partite entanglement of a 3-quDit state."""

    # pylint: enable=invalid-name


class BackendCfg(CommonBaseModel):
    """Container class grouping configuration of backend used by Gilbert algorithm."""

    name: Backend
    """Name of backend to use."""

    precision: Precision
    """Specify precision of calculations."""


class Backend(CaseInsensitiveEnum):
    """Backend to use for calculations."""

    # pylint: disable=invalid-name

    NumPy = "NumPy"

    # pylint: enable=invalid-name


class Precision(CaseInsensitiveEnum):
    """Precision of calculations performed."""

    # pylint: disable=invalid-name

    DOUBLE = "double"
    """64 bit floating point real part with 64 bit floating point complex value."""

    SINGLE = "single"
    """32 bit floating point real part with 32 bit floating point complex value."""

    # pylint: enable=invalid-name


class State(CommonBaseModel):
    """State configuration."""

    file: str
    """Path to file containing state matrix."""

    depth: Optional[int] = Field(default=None)
    """Depth of system, ie.

    number of dimensions in qu(D)it. (d)
    """

    quantity: Optional[int] = Field(default=None)
    """Quantity of systems.

    ie. number of qu(D)its in state. (n)
    """

    def eval_dynamic(self, project: CSSFProject, task_name: str, task: Task) -> None:
        """Evaluate dynamic path expressions."""
        self.file = self.file.format(project=project, task_name=task_name, task=task)


class RuntimeCfg(CommonBaseModel):
    """Configuration of runtime limits and parameters influencing algorithm run time."""

    visibility: float = Field(ge=0.0, le=1.0)
    """Visibility against white noise.

    Between 0 and 1.
    """

    max_epochs: int = Field(ge=1, le=1_000_000_000)
    """Maximal number of algorithm epochs to perform.

    If other interruption condition is met before the number of epochs, algorithm wont
    execute the rest of epochs.
    """

    iters_per_epoch: int = Field(ge=1, le=1_000_000_000)
    """Number of iterations per epochs.

    Between iterations no checks are performed, which may speed up calculations. However
    intermediate state of systems are not saved anywhere.
    """

    max_corrections: int
    """Maximal number of corrections to collect.

    Use -1 to disable this limit.
    """


class Resources(CommonBaseModel):
    """Project resources."""

    symmetries: Optional[list[str]] = Field(default=None)
    """List of paths to files containing symmetry matrices."""

    projection: Optional[str] = Field(default=None)
    """Path to file containing projection matrix."""

    def eval_dynamic(self, project: CSSFProject, task_name: str, task: Task) -> None:
        """Evaluate dynamic path expressions."""
        if self.symmetries is not None:
            for i, sym in enumerate(self.symmetries):
                self.symmetries[i] = sym.format(
                    project=project, task_name=task_name, task=task
                )

        if self.projection is not None:
            self.projection = self.projection.format(
                project=project, task_name=task_name, task=task
            )


BackendCfg.update_forward_refs()
GilbertCfg.update_forward_refs()  # type: ignore
Resources.update_forward_refs()  # type: ignore
Meta.update_forward_refs()  # type: ignore
RuntimeCfg.update_forward_refs()  # type: ignore
Task.update_forward_refs()  # type: ignore
CSSFProject.update_forward_refs()  # type: ignore
