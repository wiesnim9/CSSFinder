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
import webbrowser
from dataclasses import dataclass

import click
import pendulum

import cssfinder
from cssfinder.algorithm.gilbert import SaveCorrectionsHookError, SaveStateHookError
from cssfinder.api import AmbiguousTaskKeyError, create_report_from, run_project_from
from cssfinder.cssfproject import (
    InvalidCSSFProjectContentError,
    MalformedProjectFileError,
    ProjectFileNotFoundError,
)
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
    logging.info("CSSFinder started at %r", pendulum.now())

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


@main.group("project")
@click.pass_context
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
def _project(ctx: click.Context, path: str) -> None:
    ctx.obj.project_path = path


@_project.command("run")
@click.option(
    "--tasks",
    "-t",
    multiple=True,
    help="Use to specify names of tasks to run. When omitted, all tasks are executed.",
)
@click.pass_obj
def _run(ctx: Ctx, tasks: list[str] | None) -> None:
    """Run tasks from the project."""
    if not tasks:
        tasks = None

    if ctx.project_path is None:
        reason = "ctx.project_path shall not be None."
        raise RuntimeError(reason)

    try:
        run_project_from(ctx.project_path, tasks, is_debug=ctx.is_debug)

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


@_project.command("task-report")
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
                webbrowser.open(url=f"file:///{report.default_dest.as_posix()}")

    except AmbiguousTaskKeyError as exc:
        logging.critical(exc.args[0])
        raise SystemExit(304_00) from exc


def _log_exit(code: int) -> None:
    logging.exception("Exit with code code.")
    raise SystemExit(code)
