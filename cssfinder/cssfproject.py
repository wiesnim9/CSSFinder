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
"""


from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

import jsonref
from pydantic import ConstrainedStr, EmailStr, Field, validator

from cssfinder.base_model import CommonBaseModel
from cssfinder.enums import CaseInsensitiveEnum

if TYPE_CHECKING:
    from typing_extensions import Self

PROJECT_FILE_NAME: str = "cssfproject.json"


class CSSFProject(CommonBaseModel):
    """CSSFProject file specification."""

    meta: Meta
    """Project meta information like name and author."""

    tasks: Dict[str, Task]
    """List of tasks within project which can be executed."""

    _project_path: Path
    """Path to cssfproject.json file."""

    def __init__(
        self,
        meta: Meta,
        tasks: list[Task] | dict[str, Task],
        project_path: str | Path,
        *_: Any,
        **_k: Any,
    ) -> None:
        """Initialize instance.

        extra args and kwargs are ignored.

        """
        super().__init__(meta=meta, tasks=tasks)
        if not isinstance(project_path, Path):
            self._project_path = Path(project_path).expanduser().resolve()

        assert isinstance(project_path, Path)
        self._project_path = project_path.expanduser().resolve()

        self.bind()

    @validator("tasks", pre=True, always=True)
    @classmethod
    def _validate_tasks(
        cls,
        value: Iterable[dict[str, Any] | Task] | dict[str, dict[str, Any] | Task | Any],
    ) -> dict[str, dict[str, Any] | Task]:
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(v, (dict, Task)):
                    error_message = f"Incorrect format of Tasks field {k!r}."
                    raise IncorrectFormatOfTaskFieldError(error_message)

            return {str(k): dict(v) for k, v in value.items()}

        for i, v in enumerate(value):
            if not isinstance(v, (dict, Task)):
                error_message = f"Incorrect format of Tasks field {i!r}."
                raise IncorrectFormatOfTaskFieldError(error_message)

        return {str(i): t for i, t in enumerate(value)}

    def bind(self) -> None:
        """Bind fields to this CSSFProject object."""
        self.meta.bind(self)

        for task_name, task in self.tasks.items():
            task.bind(self, task_name)

    @property
    def project_file(self) -> Path:
        """Path to `cssfproject.json` file."""
        if self._project_path.name == PROJECT_FILE_NAME:
            return self._project_path

        return self._project_path / PROJECT_FILE_NAME

    @property
    def project_directory(self) -> Path:
        """Path to directory containing `cssfproject.json` file."""
        return self.project_file.parent

    @property
    def project_output_directory(self) -> Path:
        """Path to output directory for this project."""
        directory = self.project_directory / "output"
        directory.mkdir(0o764, parents=True, exist_ok=True)
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
            file_or_directory /= PROJECT_FILE_NAME
            project_path = file_or_directory

        else:
            project_path = file_or_directory

        logging.debug("Resolved project path to %r", project_path.as_posix())
        try:
            content = project_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            error_message = f"Make sure you path is correct: {project_path!r}"
            raise ProjectFileNotFoundError(error_message) from exc

        try:
            content = jsonref.loads(content)
        except json.decoder.JSONDecodeError as exc:
            raise MalformedProjectFileError(exc.msg, exc.doc, exc.pos) from exc

        if not isinstance(content, dict):
            logging.critical("Content of cssfproject.json file is not a dictionary.")
            raise InvalidCSSFProjectContentError(content)

        project = cls(**content, project_path=project_path)
        return project

    def select_tasks(self, patterns: list[str] | None = None) -> list[Task]:
        """Select all tasks matching list of patterns."""
        if patterns is None:
            return list(self.tasks.values())

        keys = set()

        for pattern in patterns:
            keys.update(fnmatch.filter(self.tasks.keys(), pattern))

        return [self.tasks[k] for k in keys]


class InvalidCSSFProjectContentError(ValueError):
    """Raised by load_from() when file content is not a dictionary."""


class IncorrectFormatOfTaskFieldError(ValueError):
    """Raised when "tasks" field contains incorrectly specified tasks."""


class MalformedProjectFileError(json.decoder.JSONDecodeError):
    """Rased when project file content can't be correctly decoded."""

    def __str__(self) -> str:
        """Convert exception to readable error explanation."""
        p = " " * 4
        line_index = self.lineno

        start_index = line_index - 10
        if start_index < 0:
            start_index = 0

        lines = self.doc.split("\n")[start_index:line_index]
        lines_joined = f"{p}\n" + "\n".join(
            f"{start_index + i + 1:>4}|{p}{line}" for i, line in enumerate(lines)
        )
        ellipsis_line = f"{start_index:>4}|{p}..." if start_index != 0 else ""

        context_header = f"\n\n{ellipsis_line}{lines_joined}"
        pointer_line = f"{' ' * 4}{self.colno * ' '}~~~~^^^^^"

        msg = f"{self.msg}: line {self.lineno} column {self.colno} (char {self.pos})"
        return f"{context_header}\n{pointer_line}\n{p}{msg}.\n"


class ProjectFileNotFoundError(FileNotFoundError):
    """Raised when project file can't be found in expected place."""


class _ProjectFieldMixin:
    """Mixin class for CSSFProject CommonBaseModel based fields."""

    _project: Optional[CSSFProject] = None
    """Reference to project object."""

    @property
    def project(self) -> CSSFProject:
        """Get project owning this task."""
        if self._project is None:
            raise NotBoundToProjectError(self, "Access to 'project' property.")
        return self._project

    @property
    def project_file(self) -> Path:
        """Path to `cssfproject.json` file."""
        if self._project is None:
            raise NotBoundToProjectError(self, "Access to 'project_file' property.")
        return self.project.project_file

    @property
    def project_directory(self) -> Path:
        """Path to directory containing `cssfproject.json` file."""
        if self._project is None:
            raise NotBoundToProjectError(
                self, "Access to 'project_directory' property."
            )
        return self.project.project_directory

    @property
    def project_output_directory(self) -> Path:
        """Path to output directory for this project."""
        if self._project is None:
            raise NotBoundToProjectError(
                self, "Access to 'project_output_directory' property."
            )
        return self.project.project_output_directory

    def bind(self, project: CSSFProject) -> None:
        """Bind object to specific CSSFProject."""
        self._project = project


class NotBoundToProjectError(Exception):
    """Raised when unbound object is used in context requiring it to be bound to
    CSSFProject instance.
    """

    def __init__(self, ob: Any, context_msg: str) -> None:
        super().__init__(
            f"Attempted to use unbound object {ob} in context requiring it to be "
            f"bound. ({context_msg})"
        )


class Meta(CommonBaseModel, _ProjectFieldMixin):
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

    _project: Optional[CSSFProject] = None
    """Reference to project object."""


class SemVerStr(ConstrainedStr):
    """Semantic versioning string regex, see https://semver.org/."""

    regex = (
        r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-]"
        r"[0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+("
        r"[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )


class NotBoundToTaskError(NotBoundToProjectError):
    """Raised when unbound object is used in context requiring it to be bound to Task
    instance.
    """


class _TaskMixin(_ProjectFieldMixin):
    """Mixin specifying binding interface for Task object."""

    _task_name: Optional[str] = None
    """Name of task assigned to it in project."""

    @property
    def task_output_directory(self) -> Path:
        """Path to output directory of task."""
        if self._task_name is None:
            raise NotBoundToTaskError(
                self, "Access to 'task_output_directory' property."
            )
        return self.project.project_output_directory / self.task_name

    @property
    def output_state_file(self) -> Path:
        """Path to output state file."""
        return self.task_output_directory / "state.mtx"

    @property
    def output_corrections_file(self) -> Path:
        """Path to output corrections file."""
        return self.task_output_directory / "corrections.json"

    @property
    def task_name(self) -> str:
        """Name of this task in project."""
        if self._task_name is None:
            raise NotBoundToTaskError(self, "Access to 'task_name' property.")
        return self._task_name

    def bind(
        self,
        project: CSSFProject,
        task_name: Optional[str] = None,
    ) -> None:
        """Bind object to specific Task."""
        super().bind(project)
        self._task_name = task_name


class Task(CommonBaseModel, _TaskMixin):
    """Container representing CSSFinder task with some algorithm."""

    gilbert: Optional[GilbertCfg] = Field(default=None)
    """Configuration of gilbert algorithm."""

    _project: Optional[CSSFProject] = None
    """Reference to project object."""

    _task_name: Optional[str] = None
    """Name of task assigned to it in project."""

    def bind(self, project: CSSFProject, task_name: Optional[str] = None) -> None:
        """Bind task to specific CSSFProject instance."""
        super().bind(project, task_name)
        if self.gilbert is not None:
            self.gilbert.bind(project, task_name, self)


class _TaskFieldMixin(_TaskMixin):
    """Mixin specifying binding interface for Task field."""

    _task: Optional[Task] = None
    """Reference to task object containing this object."""

    @property
    def task(self) -> Task:
        """Name of this task in project."""
        if self._task is None:
            raise NotBoundToTaskError(self, "Access to 'task' property.")
        return self._task

    def bind(
        self,
        project: CSSFProject,
        task_name: Optional[str] = None,
        task: Optional[Task] = None,
    ) -> None:
        """Evaluate dynamic fields of CSSFProject element."""
        super().bind(project, task_name)
        self._task = task


class GilbertCfg(CommonBaseModel, _TaskFieldMixin):
    """Gilbert algorithm configuration container class."""

    mode: AlgoMode
    """Algorithm mode to use."""

    backend: Optional[BackendCfg] = Field(default=None)
    """Configuration of backend which will be used for execution.

    When backend configuration is not specified, numpy with double precision is used.

    """

    state: Union[State, str]
    """Path to file containing initial state matrix."""

    runtime: RuntimeCfg
    """Configuration of runtime limits and parameters influencing algorithm run time."""

    resources: Optional[Resources] = Field(default=None)
    """Additional resources which may be used by algorithm."""

    _project: Optional[CSSFProject] = None
    """Reference to project object."""

    _task_name: Optional[str] = None
    """Name of task assigned to it in project."""

    _task: Optional[Task] = None
    """Reference to task object containing this object."""

    @validator("state", always=True)
    @classmethod
    def _validate_state(cls, value: str | State) -> State:
        if not isinstance(value, State):
            return State(file=value)

        return value

    def get_backend(self) -> BackendCfg:
        """Return resources object."""
        if self.backend is None:
            self.backend = BackendCfg(name=Backend.NumPy, precision=Precision.DOUBLE)
        return self.backend

    def get_resources(self) -> Resources:
        """Return resources object."""
        if self.resources is None:
            self.resources = Resources()
        return self.resources

    def bind(
        self,
        project: CSSFProject,
        task_name: Optional[str] = None,
        task: Optional[Task] = None,
    ) -> None:
        """Evaluate dynamic path expressions."""
        super().bind(project, task_name, task)
        assert isinstance(self.state, State)

        self.state.bind(project, task_name, task)
        self.get_resources().bind(project, task_name, task)

    def get_state(self) -> State:
        """Return initial state information."""
        assert isinstance(self.state, State)
        return self.state


class AlgoMode(CaseInsensitiveEnum):
    """Mode of algorithm."""

    # pylint: disable=invalid-name

    FSnQd = "FSnQd"
    """Full separability of n-quDit state."""

    SBiPa = "SBiPa"
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


class State(CommonBaseModel, _TaskFieldMixin):
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

    _project: Optional[CSSFProject] = None
    """Reference to project object."""

    _task_name: Optional[str] = None
    """Name of task assigned to it in project."""

    _task: Optional[Task] = None
    """Reference to task object containing this object."""

    def bind(
        self,
        project: CSSFProject,
        task_name: Optional[str] = None,
        task: Optional[Task] = None,
    ) -> None:
        """Evaluate dynamic path expressions.

        Path expands user (~) and is resolved only when correctly bound to project.

        """
        super().bind(project, task_name, task)

        if task_name is None or task is None:
            return

        self.file = (
            Path(self.file.format(project=project, task_name=task_name, task=task))
            .expanduser()
            .resolve()
            .as_posix()
        )

    def is_predefined_dimensions(self) -> bool:
        """Return True when both dimensions are available."""
        return self.depth is not None and self.quantity is not None

    def get_depth(self) -> int:
        """Return system depth or raise NoDimensionsError if not specified in config."""
        if self.depth is None:
            msg = "Depth is not specified."
            raise NoDimensionsError(msg)
        return self.depth

    def get_quantity(self) -> int:
        """Return system quantity or raise NoDimensionsError if not specified in
        config.
        """
        if self.quantity is None:
            msg = "quantity is not specified."
            raise NoDimensionsError(msg)
        return self.quantity


class NoDimensionsError(ValueError):
    """Raised when system dimensions were requested but are not specified in config."""


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


class Resources(CommonBaseModel, _TaskFieldMixin):
    """Project resources."""

    symmetries: Optional[List[List[str]]] = Field(default=None)
    """List of paths to files containing symmetry matrices."""

    projection: Optional[str] = Field(default=None)
    """Path to file containing projection matrix."""

    _project: Optional[CSSFProject] = None
    """Reference to project object."""

    _task_name: Optional[str] = None
    """Name of task assigned to it in project."""

    _task: Optional[Task] = None
    """Reference to task object containing this object."""

    def bind(
        self,
        project: CSSFProject,
        task_name: Optional[str] = None,
        task: Optional[Task] = None,
    ) -> None:
        """Evaluate dynamic path expressions.

        Paths expands user (~) and are resolved only when correctly bound to project.

        """
        super().bind(project, task_name, task)

        if task_name is None or task is None:
            return

        if self.symmetries is not None:
            self.symmetries = [
                [
                    Path(
                        sym.format(
                            project=project,
                            task_name=task_name,
                            task=task,
                        )
                    )
                    .expanduser()
                    .resolve()
                    .as_posix()
                    for sym in row
                ]
                for row in self.symmetries
            ]

        if self.projection is not None:
            self.projection = (
                Path(
                    self.projection.format(
                        project=project,
                        task_name=task_name,
                        task=task,
                    )
                )
                .expanduser()
                .resolve()
                .as_posix()
            )


BackendCfg.update_forward_refs()
GilbertCfg.update_forward_refs()
Resources.update_forward_refs()
Meta.update_forward_refs()
RuntimeCfg.update_forward_refs()
Task.update_forward_refs()
CSSFProject.update_forward_refs()
