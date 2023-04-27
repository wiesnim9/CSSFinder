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


"""Collection of functions making up interactive interface of CSSFinder."""

from __future__ import annotations

import getpass
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytermgui as ptg
from pydantic import EmailStr, ValidationError

from cssfinder.cssfproject import (
    AlgoMode,
    BackendCfg,
    CSSFProject,
    GilbertCfg,
    Meta,
    Precision,
    RuntimeCfg,
    SemVerStr,
    State,
    Task,
)


def create_new_project(
    default_author: Optional[str] = None,
    default_email: Optional[str] = None,
    default_name: Optional[str] = None,
    default_description: Optional[str] = None,
    default_version_string: Optional[str] = None,
) -> None:
    """Create new project directory and cssfinder.json file."""
    all_set = (
        default_author is not None
        and default_email is not None
        and default_name is not None
        and default_description is not None
        and default_version_string is not None
    )

    author = default_author or _load_default_name_from_git()
    email = default_email or _load_default_email_from_git()
    name = default_name or "new_project"
    description = default_description or " "
    version_string = default_version_string or "1.0.0"

    if all_set:
        meta = Meta(
            author=author,
            email=EmailStr(email),
            name=name,
            description=description,
            version=SemVerStr(version_string),
        )
    else:
        meta = get_project_fields_with_pytermgui(
            author,
            email,
            name,
            description,
            version_string,
        )

    project_file_path = Path.cwd() / meta.name / "cssfproject.json"
    if project_file_path.exists():
        if (
            input("Project already exists, override? (y/n) ").casefold()
            == "Y".casefold()
        ):
            project_file_path.unlink()
        else:
            print("Aborted.")
            raise SystemExit(1)

    project_file_path.parent.mkdir(0o777, parents=True, exist_ok=True)
    project_file_path.touch(0o777, exist_ok=True)

    project = CSSFProject(
        meta=meta,
        tasks=[],
        project_path=project_file_path.as_posix(),
    )
    serialized = project.json(indent=4, ensure_ascii=False)
    project_file_path.write_text(serialized)


def _load_default_name_from_git() -> str:
    # Retrieve default from system name just in case, but git usually contains
    # better value.
    default_name = getpass.getuser()

    try:
        retval = subprocess.run(["git", "config", "user.name"], capture_output=True)
        if retval.returncode == 0:
            default_name = retval.stdout.decode("utf-8").strip()

    except (FileNotFoundError, ValueError):
        pass

    return default_name


def _load_default_email_from_git() -> EmailStr:
    default_email = EmailStr("unknown@unknown.com")

    try:
        retval = subprocess.run(["git", "config", "user.email"], capture_output=True)
        if retval.returncode == 0:
            default_email = EmailStr(retval.stdout.decode("utf-8").strip())

    except (FileNotFoundError, ValueError, ValidationError):
        pass

    return default_email


def get_project_fields_with_pytermgui(
    default_author_name: str,
    default_author_email: str,
    default_project_name: str,
    default_project_description: str,
    default_project_version: str,
) -> Meta:
    """Request user for meta values for project using TUI."""
    message = ""

    while True:
        (
            default_author_name,
            default_author_email,
            default_project_name,
            default_project_description,
            default_project_version,
        ) = _get_project_fields_with_pytermgui(
            default_author_name,
            default_author_email,
            default_project_name,
            default_project_description,
            default_project_version,
            message,
        )

        try:
            return Meta(
                author=default_author_name,
                email=EmailStr(default_author_email),
                name=default_project_name,
                description=default_project_description,
                version=SemVerStr(default_project_version),
            )
        except ValidationError as e:
            message = f"[210 bold]{e}"


def _get_project_fields_with_pytermgui(  # noqa: PLR0913
    default_author_name: str,
    default_author_email: str,
    default_project_name: str,
    default_project_description: str,
    default_project_version: str,
    message: str,
) -> tuple[str, str, str, str, str]:
    is_interrupted: bool = True

    with ptg.WindowManager() as manager:

        def _cb(_widget: ptg.Button) -> None:
            nonlocal is_interrupted
            is_interrupted = False
            manager.stop()

        window = ptg.Window(
            "[63 bold]Project metadata",
            *(() if not message else (message,)),
            "",
            # fmt: off
            # ruff: noqa: E501
            (author_name := InputField(default_author_name, prompt="Author Name: ")),
            (author_email := InputField(default_author_email, prompt="Author Email: ")),
            (project_name := InputField(default_project_name, prompt="Project Name: ")),
            (project_description := InputField(default_project_description, prompt="Project Description: ")),
            (project_version := InputField(default_project_version, prompt="Project Version: ")),
            # ruff: noqa: E501
            # fmt: on
            "",
            ptg.Container(
                ptg.Button(
                    "Submit",
                    onclick=_cb,
                ),
                box="EMPTY_VERTICAL",
            ),
            width=60,
            box="DOUBLE",
        ).center()

        manager.add(window)

    if is_interrupted:
        raise KeyboardInterrupt

    return (
        author_name.value,
        author_email.value,
        project_name.value,
        project_description.value,
        project_version.value,
    )


@dataclass
class GilbertTaskSpec:
    """Specification of task."""

    name: str
    mode: str
    backend_name: str
    precision: str
    state: Optional[str] = None
    depth: Optional[str] = None
    quantity: Optional[str] = None
    visibility: Optional[str] = None
    max_epochs: Optional[str] = None
    iters_per_epoch: Optional[str] = None
    max_corrections: Optional[str] = None
    symmetries: Optional[str] = None
    projection: Optional[str] = None
    derive: Optional[str] = None

    def to_task(self) -> Task:
        """Create Task object with values from spec."""
        if self.state is None:
            msg = "Path to file containing state matrix must be specified."
            raise ValueError(msg)

        return Task(
            gilbert=GilbertCfg(
                mode=AlgoMode(self.mode),
                backend=BackendCfg(
                    name=self.backend_name,  # type: ignore[arg-type]
                    precision=Precision(self.precision),
                ),
                state=State(
                    file=self.state,  # type: ignore[arg-type]
                    depth=self.depth,  # type: ignore[arg-type]
                    quantity=self.quantity,  # type: ignore[arg-type]
                ),
                runtime=RuntimeCfg(
                    visibility=self.visibility,  # type: ignore[arg-type]
                    max_epochs=self.max_epochs,  # type: ignore[arg-type]
                    iters_per_epoch=self.iters_per_epoch,  # type: ignore[arg-type]
                    max_corrections=self.max_corrections,  # type: ignore[arg-type]
                ),
            ),
        )


def add_task_gilbert(
    project: CSSFProject,
    spec: GilbertTaskSpec,
) -> None:
    """Add task to project and save it in place."""
    while True:
        try:
            task = spec.to_task()
            break

        except (ValueError, TypeError, ValidationError, KeyError):
            spec = get_gilbert_task_fields_with_pytermgui(spec)

    if spec.name in project.tasks and (
        input("Task already exists, override? (y/n) ").casefold() != "Y".casefold()
    ):
        print("Aborted.")
        raise SystemExit(1)

    project.tasks[spec.name] = task
    project.project_file.write_text(project.json(indent=4, ensure_ascii=False))


class InputField(ptg.InputField):
    """Custom Input field which avoids styling value."""

    def _style_and_break_lines(self) -> list[str]:
        """Styles and breaks self._lines."""
        document = (self.styles.prompt(self.prompt) + self.value).splitlines()

        lines: list[str] = []
        width = self.width
        extend = lines.extend

        for line in document:
            extend(ptg.break_line(line.replace("\n", "\\n"), width, fill=" "))
            extend("")

        return lines


def get_gilbert_task_fields_with_pytermgui(
    spec: GilbertTaskSpec, message: Optional[str] = None
) -> GilbertTaskSpec:
    """Create temporary TUI prompt for entering task configuration."""
    is_interrupted: bool = True
    df_state = "{project.project_directory}/state.mtx"

    with ptg.WindowManager() as manager:

        def _cb(_widget: ptg.Button) -> None:
            nonlocal is_interrupted
            is_interrupted = False
            manager.stop()

        manager.layout = ptg.Layout()
        manager.layout.add_slot("body", width=60)

        # fmt: off
        # ruff: noqa: E501
        manager.add(
            ptg.Window(
                ptg.Container(
                    "[117 bold]New Task",
                    *(() if not message else (message,)),
                    "",
                    ptg.Container(
                (field_name := InputField(spec.name, prompt="Task Name*: ")),
                    ),
                    "",
                ptg.Label("[!gradient(33)]Backend", parent_align=0),
                    ptg.Container(
                (field_backend_name := InputField(spec.backend_name, prompt="Backend Name: ")),
                (field_precision := InputField(spec.precision, prompt="Precision: ")),
                    ),
                ptg.Label("[!gradient(63)]Initial System State", parent_align=0),
                    ptg.Container(
                (field_state := InputField(spec.state or df_state or "?", prompt="State File Path*: ")),
                (field_depth := InputField(spec.depth or "None", prompt="System Depth: ")),
                (field_quantity := InputField(spec.quantity or "None", prompt="System Quantity: ")),
                    ),
                ptg.Label("[!gradient(63)]Execution Rules", parent_align=0),
                    ptg.Container(
                (field_mode := InputField(spec.mode, prompt="Task Mode*: ")),
                (field_visibility := InputField(spec.visibility or "", prompt="Visibility: ")),
                (field_max_epochs := InputField(spec.max_epochs or "", prompt="Max Epochs: ")),
                (field_iters_per_epoch := InputField(spec.iters_per_epoch or "", prompt="Iters Per Epoch: ")),
                (field_max_corrections := InputField(spec.max_corrections or "", prompt="Max Corrections: ")),
                    ),
                ptg.Label("[!gradient(63)]Modifiers", parent_align=0),
                    ptg.Container(
                (field_symmetries := InputField(spec.symmetries or "None", prompt="Symmetries: ")),
                (field_projection := InputField(spec.projection or "None", prompt="Projection: ")),
                    ),
                    "",
                    "* - Field is required.",
                    ptg.Container(
                        ptg.Button("Submit", onclick=_cb),
                        box=ptg.boxes.EMPTY_VERTICAL,
                    ),
                    box=ptg.boxes.DOUBLE,
                ),
                vertical_align=ptg.VerticalAlignment.TOP,
                overflow=ptg.Overflow.SCROLL,
                box=ptg.boxes.EMPTY,
            )
            .center(),
            assign="body",
        )
        # ruff: noqa: E501
        # fmt: on

    if is_interrupted:
        raise KeyboardInterrupt

    spec.name = field_name.value
    spec.mode = field_mode.value
    spec.backend_name = field_backend_name.value
    spec.precision = field_precision.value

    def to_none_if_literal(some: str) -> Optional[str]:
        if some.strip().casefold() in (
            "none".casefold(),
            "null".casefold(),
        ):
            return None
        return some

    spec.state = field_state.value
    spec.depth = to_none_if_literal(field_depth.value)
    spec.quantity = to_none_if_literal(field_quantity.value)

    spec.visibility = field_visibility.value
    spec.max_epochs = field_max_epochs.value
    spec.iters_per_epoch = field_iters_per_epoch.value
    spec.max_corrections = field_max_corrections.value

    spec.symmetries = field_symmetries.value
    spec.projection = field_projection.value

    return spec
