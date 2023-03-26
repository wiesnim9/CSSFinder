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
from pathlib import Path
from typing import Optional

import pytermgui as ptg
from pydantic import EmailStr, ValidationError

from cssfinder.cssfproject import CSSFProject, Meta, SemVerStr


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
            author_name,
            author_email,
            project_name,
            project_description,
            project_version,
        ) = _run_tui(
            default_author_name,
            default_author_email,
            default_project_name,
            default_project_description,
            default_project_version,
            message,
        )

        try:
            return Meta(
                author=author_name,
                email=EmailStr(author_email),
                name=project_name,
                description=project_description,
                version=SemVerStr(project_version),
            )
        except ValidationError as e:
            message = f"[210 bold]{e}"


def _run_tui(  # noqa: PLR0913
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
            (
                author_name := ptg.InputField(
                    default_author_name, prompt="Author Name: "
                )
            ),
            (
                author_email := ptg.InputField(
                    default_author_email, prompt="Author Email: "
                )
            ),
            (
                project_name := ptg.InputField(
                    default_project_name, prompt="Project Name: "
                )
            ),
            (
                project_description := ptg.InputField(
                    default_project_description, prompt="Project Description: "
                )
            ),
            (
                project_version := ptg.InputField(
                    default_project_version, prompt="Project Version: "
                )
            ),
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
