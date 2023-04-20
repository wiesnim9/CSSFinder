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
from typing import TYPE_CHECKING, Callable, Optional

import click
import pendulum
import rich

import cssfinder
from cssfinder.log import enable_performance_logging

if TYPE_CHECKING:
    from cssfinder import examples

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
@click.option("--perf-log", is_flag=True, default=False)
def main(ctx: click.Context, verbose: int, *, debug: bool, perf_log: bool) -> None:
    """CSSFinder is a script for finding closest separable states."""
    from cssfinder.log import configure_logger

    configure_logger(verbosity=verbose, logger_name="cssfinder", use_rich=False)
    ctx.obj = Ctx(is_debug=debug)

    logging.getLogger("numba").setLevel(logging.ERROR)
    logging.info("CSSFinder started at %s", pendulum.now().isoformat(sep=" "))

    if perf_log:
        enable_performance_logging()

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
    from cssfinder.interactive import create_new_project

    create_new_project(author, email, name, description, project_version)


@_project.group("task")
def _task() -> None:
    """Group of commands to operate on tasks."""


@_task.group("add")
def _add() -> None:
    """Command for adding new tasks."""


@_add.command("gilbert")
@click.pass_obj
@click.option("--name", default=None, help="Name for the task.")
@click.option("--mode", default=None, help="Algorithm mode.")
@click.option(
    "--backend-name",
    default=None,
    help="Name of backend. Use `cssfinder backend list` to show installed backends.",
)
@click.option("--precision", default=None, help="Precision of calculations.")
@click.option(
    "--state",
    default=None,
    help="Path to matrix file containing initial system state.",
)
@click.option(
    "--depth",
    default=None,
    help="Depth of system, ie. number of dimensions in qu(D)it. (d)",
)
@click.option(
    "--quantity",
    default=None,
    help="Quantity of systems. ie. number of qu(D)its in state. (n)",
)
@click.option(
    "--visibility",
    default=None,
    help="Visibility against white noise, Between 0 and 1.",
)
@click.option(
    "--max-epochs", default=None, help="Maximal number of algorithm epochs to perform."
)
@click.option(
    "--iters-per-epoch", default=None, help="Number of iterations per single epoch."
)
@click.option(
    "--max-corrections",
    default=None,
    help="Maximal number of corrections to collect. Because halt condition is checked "
    "once per epoch, number of total corrections might exceed this limit for long "
    "epochs. Use -1 to disable this limit.",
)
@click.option(
    "--derive",
    default=None,
    help="Declare name of other existing task to derive missing field values from.",
)
@click.option(
    "--symmetries",
    default=None,
    help="List of lists of files containing symmetries matrices as valid JSON literal.",
)
@click.option(
    "--projection",
    default=None,
    help="Path to file containing projection matrix.",
)
def _gilbert(  # noqa: PLR0913
    ctx: Ctx,
    name: Optional[str],
    mode: Optional[str],
    backend_name: Optional[str],
    precision: Optional[str],
    state: Optional[str],
    depth: Optional[str],
    quantity: Optional[str],
    visibility: Optional[str],
    max_epochs: Optional[str],
    iters_per_epoch: Optional[str],
    max_corrections: Optional[str],
    symmetries: Optional[str],
    projection: Optional[str],
    derive: Optional[str],
) -> None:
    """Add new gilbert algorithm task.

    Task options can either be given by command line parameters or later interactively.

    """
    from cssfinder.cssfproject import CSSFProject
    from cssfinder.interactive import GilbertTaskSpec, add_task_gilbert

    if ctx.project_path is None:
        reason = "ctx.project_path shall not be None."
        raise RuntimeError(reason)

    project = CSSFProject.load_project(ctx.project_path)

    add_task_gilbert(
        project,
        GilbertTaskSpec(
            name or f"task_{len(project.tasks)}",
            mode or "FSnQd",
            backend_name or "numpy_jit",
            precision or "single",
            state,
            depth,
            quantity,
            visibility or "0.4",
            max_epochs or "100",
            iters_per_epoch or "10000",
            max_corrections or "1000",
            symmetries or "[]",
            projection,
            derive,
        ),
    )


@_task.command("run")
@click.option(
    "--match",
    "-m",
    "match_",
    multiple=True,
    help="Use to specify names of tasks to run. When omitted, all tasks are executed.",
)
@click.option(
    "--force-sequential",
    is_flag=True,
    default=False,
    help="Enforce sequential execution. As opposed to --max-parallel set to 1, "
    "this causes code to execute only in main thread.",
)
@click.option(
    "--max-parallel",
    "-p",
    type=int,
    default=-1,
    help="Limit maximal number of tasks executed in parallel. Note that this never "
    "changes execution scheme, thus code won't be executed in main thread.",
)
@click.pass_obj
def _run(
    ctx: Ctx, match_: list[str] | None, *, force_sequential: bool, max_parallel: int
) -> None:
    """Run tasks from the project."""
    from cssfinder.algorithm.gilbert import SaveCorrectionsHookError, SaveStateHookError
    from cssfinder.api import run_project_from
    from cssfinder.cssfproject import (
        InvalidCSSFProjectContentError,
        MalformedProjectFileError,
        ProjectFileNotFoundError,
    )

    if not match_:
        match_ = None

    if ctx.project_path is None:
        reason = "ctx.project_path shall not be None."
        raise RuntimeError(reason)

    try:
        run_project_from(
            ctx.project_path,
            match_,
            is_debug=ctx.is_debug,
            force_sequential=force_sequential,
            max_parallel=max_parallel,
        )

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
    from cssfinder.api import AmbiguousTaskKeyError, create_report_from
    from cssfinder.reports.renderer import ReportType

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
    from cssfinder.algorithm.backend.loader import Loader

    rich.get_console().print(Loader.new().get_rich_table())


@main.group("examples")
def _examples() -> None:
    """Group of commands for accessing bundled examples."""


@_examples.command("list")
def _examples_list() -> None:
    """Show list of all available example projects."""
    from cssfinder import examples

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
    from cssfinder.crossplatform import open_file_explorer, open_terminal
    from cssfinder.cssfproject import ProjectFileNotFoundError
    from cssfinder.enums import ExitCode

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
    from cssfinder.enums import ExitCode

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
    from cssfinder import examples
    from cssfinder.enums import ExitCode

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
