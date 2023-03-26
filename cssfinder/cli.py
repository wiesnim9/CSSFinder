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


"""Module contains implementation of CSSFinder command line interface."""

from __future__ import annotations

import logging
import shutil
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import click
import pendulum
import rich

import cssfinder
from cssfinder import examples
from cssfinder.algorithm.backend.loader import Loader
from cssfinder.algorithm.gilbert import SaveCorrectionsHookError, SaveStateHookError
from cssfinder.api import AmbiguousTaskKeyError, create_report_from, run_project_from
from cssfinder.crossplatform import open_file_explorer, open_terminal
from cssfinder.cssfproject import (
    InvalidCSSFProjectContentError,
    MalformedProjectFileError,
    ProjectFileNotFoundError,
)
from cssfinder.enums import ExitCode
from cssfinder.interactive import create_new_project
from cssfinder.log import configure_logger
from cssfinder.reports.renderer import ReportType

VERBOSITY_INFO: int = 2


@dataclass
class Ctx:
    """Command line context wrapper class."""

    is_debug: bool = False
    project_path: str | None = None


@click.group(invoke_without_command=True, no_args_is_help=True)
@click.pass_context
@click.option(
    "-v",
    "--verbose",
    default=0,
    count=True,
    help="Control verbosity of logging, by default+ critical only, use "
    "-v, -vv, -vvv to gradually increase it.",
)
@click.version_option(cssfinder.__version__, "-V", "--version", prog_name="cssfinder")
@click.option("--debug", is_flag=True, default=False)
def main(ctx: click.Context, verbose: int, *, debug: bool) -> None:
    """CSSFinder is a script for finding closest separable states."""
    ctx.obj = Ctx(is_debug=debug)

    configure_logger(verbosity=verbose, logger_name="cssfinder", use_rich=False)
    logging.getLogger("numba").setLevel(logging.ERROR)
    logging.info("CSSFinder started at %s", pendulum.now().isoformat(sep=" "))

    if verbose >= VERBOSITY_INFO:
        print(
            """
  ██████╗███████╗███████╗███████╗██╗███╗   ██╗██████╗ ███████╗██████╗
 ██╔════╝██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔══██╗██╔════╝██╔══██╗
 ██║     ███████╗███████╗█████╗  ██║██╔██╗ ██║██║  ██║█████╗  ██████╔╝
 ██║     ╚════██║╚════██║██╔══╝  ██║██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗
 ╚██████╗███████║███████║██║     ██║██║ ╚████║██████╔╝███████╗██║  ██║
  ╚═════╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝
""",
        )


def _is_path_needed_for_subcommand(
    ctx: click.Context, param: click.Option, value: Optional[str]  # noqa: ARG001
) -> str | None:
    print(type(value))
    if ctx.invoked_subcommand in ["task"] and not value:
        msg = "The path parameter is required for this subcommand."
        raise click.BadParameter(msg)
    return value


@main.group("project")
@click.pass_context
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    callback=_is_path_needed_for_subcommand,
)
def _project(ctx: click.Context, path: str) -> None:
    """Group of commands for interaction with projects."""
    ctx.obj.project_path = path


@_project.command("new")
@click.option("--author", default=None, help="Author metadata field value.")
@click.option("--email", default=None, help="Email metadata field value.")
@click.option("--name", default=None, help="Name metadata field value.")
@click.option("--description", default=None, help="Description metadata field value.")
@click.option("--project-version", default=None, help="Version metadata field value.")
def _project_new(
    author: Optional[str],
    email: Optional[str],
    name: Optional[str],
    description: Optional[str],
    project_version: Optional[str],
) -> None:
    """Create new project."""
    create_new_project(author, email, name, description, project_version)


@_project.group("task")
def _task() -> None:
    """Group of commands to operate on tasks."""


@_task.command("run")
@click.option(
    "--match",
    "-m",
    "match_",
    multiple=True,
    help="Use to specify names of tasks to run. When omitted, all tasks are executed.",
)
@click.pass_obj
def _run(ctx: Ctx, match_: list[str] | None) -> None:
    """Run tasks from the project."""
    if not match_:
        match_ = None

    if ctx.project_path is None:
        reason = "ctx.project_path shall not be None."
        raise RuntimeError(reason)

    try:
        run_project_from(ctx.project_path, match_, is_debug=ctx.is_debug)

    except ProjectFileNotFoundError as exc:
        logging.critical("Project file not found. %s", exc.args[0])
        raise SystemExit(300_000) from exc

    except MalformedProjectFileError as exc:
        logging.critical("Couldn't parse `cssfproject.json` file.")
        logging.critical(exc)
        raise SystemExit(301_000) from exc

    except InvalidCSSFProjectContentError as exc:
        logging.critical("Project file doesn't contain valid project configuration.")
        logging.critical("Fix it and try again.")
        raise SystemExit(302_000) from exc

    except SaveStateHookError:
        _log_exit(303_000)

    except SaveCorrectionsHookError:
        _log_exit(304_000)

    raise SystemExit(0)


@_task.command("report")
@click.argument(
    "task",
)
@click.option(
    "--html",
    "--no-html",
    is_flag=True,
    default=False,
    help="Include HTML report.",
)
@click.option(
    "--pdf",
    "--no-pdf",
    is_flag=True,
    default=False,
    help="Include PDF report.",
)
@click.option(
    "--open",
    "--no-open",
    "open_",
    is_flag=True,
    default=False,
    help="Automatically open report in web browser.",
)
@click.pass_obj
def _task_report(ctx: Ctx, task: str, *, html: bool, pdf: bool, open_: bool) -> None:
    """Create short report for task.

    TASK - name pattern matching exactly one task, for which report should be created.

    """
    assert ctx.project_path is not None

    include_report_types = []

    if html:
        include_report_types.append(ReportType.HTML)

    if pdf:
        include_report_types.append(ReportType.PDF)

    if len(include_report_types) == 0:
        logging.critical(
            "No report type was selected therefore nothing will be calculated, exiting."
        )
        raise SystemExit(0)

    try:
        for report in create_report_from(ctx.project_path, task, include_report_types):
            report.save_default()
            if open_:
                report.get_default_dest()
                webbrowser.open(url=report.get_default_dest().as_uri())

    except AmbiguousTaskKeyError as exc:
        logging.critical(exc.args[0])
        raise SystemExit(304_00) from exc


def _log_exit(code: int) -> None:
    logging.exception("Exit with code code.")
    raise SystemExit(code)


@main.group("backend")
def _backend() -> None:
    """Group of commands for inspecting available backends."""


@_backend.command("list")
def _backend_list() -> None:
    """List available backends."""
    rich.get_console().print(Loader.new().get_rich_table())


@main.group("examples")
def _examples() -> None:
    """Group of commands for accessing bundled examples."""


@_examples.command("list")
def _examples_list() -> None:
    """Show list of all available example projects."""
    console = rich.get_console()
    table = examples.Example.get_info_table()
    console.print()
    console.print(table)


def validate_mutually_exclusive(
    this: str,
    other: str,
) -> Callable[[click.Context, dict[str, str], str], Optional[str]]:
    """Return callback checking for mutually exclusive options."""

    def _(
        ctx: click.Context, param: dict[str, str], value: Optional[str]  # noqa: ARG001
    ) -> Optional[str]:
        if value is not None and ctx.params.get(other) is not None:
            msg = f"{this!r} and {other!r} options are mutually exclusive."
            raise click.BadParameter(msg)

        return value

    return _


@_examples.command("clone")
@click.option(
    "--sha",
    default=None,
    help="SHA of example. Mutually exclusive with `--name`.",
    expose_value=True,
)
@click.option(
    "--name",
    default=None,
    help="Name of example. Mutually exclusive with `--sha`.",
    expose_value=True,
    callback=validate_mutually_exclusive("name", "sha"),
)
@click.option(
    "-o",
    "--out",
    default=None,
    help="Path to destination directory, where project folder should be placed.",
)
@click.option(
    "-f",
    "--force",
    "force_overwrite",
    is_flag=True,
    help="Remove and recreate project folder in destination if one already exists.",
)
@click.option(
    "-t",
    "--terminal",
    "do_open_terminal",
    is_flag=True,
    help="When set, automatically open new terminal window in example directory.",
)
@click.option(
    "-e",
    "--explorer",
    "do_open_explorer",
    is_flag=True,
    help="When set, automatically open new explorer window in example directory.",
)
def _examples_clone(
    sha: Optional[str],
    name: Optional[str],
    out: Optional[str],
    *,
    force_overwrite: bool,
    do_open_terminal: bool,
    do_open_explorer: bool,
) -> None:
    """Clone one of examples to specific location."""
    destination = Path.cwd() if out is None else Path(out).expanduser().resolve()

    example = _select_example(sha, name)
    try:
        project = example.get_project()
    except ProjectFileNotFoundError as exc:
        logging.debug(traceback.format_exc())
        logging.critical(
            "Sorry but example is broken. (%s)", exc.__class__.__qualname__
        )
        raise SystemExit(ExitCode.BROKEN_EXAMPLE) from exc

    rich.print(
        f"Found example {example.name!r}, {project.meta.author!r}, "
        f"{example.get_sha256().hexdigest()[:8]!r}"
    )

    destination_project_folder = _get_validated_destination(
        destination, example, force_overwrite=force_overwrite
    )
    try:
        example.clone(destination)

    except FileNotFoundError as exc:
        logging.critical(str(exc))
        raise SystemExit(ExitCode.PROJECT_NOT_FOUND) from exc

    if do_open_explorer:
        open_file_explorer(destination_project_folder)
    if do_open_terminal:
        open_terminal(destination_project_folder)


def _get_validated_destination(
    destination: Path, example: examples.Example, *, force_overwrite: bool
) -> Path:
    destination_project_folder = destination / example.folder_name
    is_destination_exists = destination_project_folder.exists()

    try:
        is_destination_non_empty = len(list(destination_project_folder.iterdir())) > 0
    except FileNotFoundError:
        is_destination_non_empty = False

    if is_destination_exists and is_destination_non_empty:
        if force_overwrite:
            shutil.rmtree(destination_project_folder.as_posix())

        else:
            logging.critical(
                "Output directory already contains folder %r, change destination "
                "folder. Remove existing folder or use `--force` flag to remove it "
                "automatically.",
                example.folder_name,
            )
            raise SystemExit(ExitCode.EXAMPLE_DESTINATION_ALREADY_EXISTS)

    return destination_project_folder


def _select_example(sha: Optional[str], name: Optional[str]) -> examples.Example:
    if name is not None:
        try:
            example = examples.Example.select_by_name(name)
        except examples.ExampleNotFoundError as exc:
            logging.critical("%s", exc)
            raise SystemExit(ExitCode.EXAMPLE_WITH_NAME_NOT_FOUND) from exc

    elif sha is not None:
        try:
            example = examples.Example.select_by_sha256(sha)
        except examples.ExampleNotFoundError as exc:
            logging.critical("%s", exc)
            raise SystemExit(ExitCode.EXAMPLE_WITH_SHA_NOT_FOUND) from exc

    else:
        logging.critical("Neither `--name` not `--sha` given, no example was selected.")
        raise SystemExit(ExitCode.EXAMPLE_SHA_NOR_NAME_GIVEN)

    return example
