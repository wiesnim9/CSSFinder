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


"""This module contains implementation of CSSFinder command line interface."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import pendulum

import cssfinder
from cssfinder.algorithm.gilbert import SaveCorrectionsHookError, SaveStateHookError
from cssfinder.api import AmbiguousTaskKeyError, create_report_from, run_project_from
from cssfinder.cssfproject import (
    InvalidCSSFProjectContent,
    MalformedProjectFileError,
    ProjectFileNotFound,
)
from cssfinder.log import configure_logger


@dataclass
class Ctx:
    """Command line context wrapper class."""

    is_debug: bool = False
    project_path: Optional[str] = None


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
def main(ctx: click.Context, verbose: int, debug: bool) -> None:
    """CSSFinder is a script for finding closest separable states."""
    ctx.obj = Ctx(is_debug=debug)

    configure_logger(verbosity=verbose, logger_name="cssfinder", use_rich=False)
    logging.getLogger("numba").setLevel(logging.ERROR)
    logging.info("CSSFinder started at %r", pendulum.now())

    if verbose >= 2:
        print(
            """
  ██████╗███████╗███████╗███████╗██╗███╗   ██╗██████╗ ███████╗██████╗
 ██╔════╝██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔══██╗██╔════╝██╔══██╗
 ██║     ███████╗███████╗█████╗  ██║██╔██╗ ██║██║  ██║█████╗  ██████╔╝
 ██║     ╚════██║╚════██║██╔══╝  ██║██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗
 ╚██████╗███████║███████║██║     ██║██║ ╚████║██████╔╝███████╗██║  ██║
  ╚═════╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝
"""
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
def _run(ctx: Ctx, tasks: Optional[list[str]]) -> None:
    """Run tasks from the project."""

    if not tasks:
        tasks = None

    if ctx.project_path is None:
        raise RuntimeError("ctx.project_path shall not be None.")

    try:
        run_project_from(ctx.project_path, tasks, is_debug=ctx.is_debug)

    except ProjectFileNotFound as exc:
        logging.critical("Project file not found. %s", exc.args[0])
        raise SystemExit(300_000) from exc

    except MalformedProjectFileError as exc:
        logging.critical(
            "Project file content is not a valid JSON file. Fix it and try again."
        )
        raise SystemExit(301_000) from exc

    except InvalidCSSFProjectContent as exc:
        logging.critical("Project file doesn't contain valid project configuration.")
        logging.critical("Fix it and try again.")
        raise SystemExit(302_000) from exc

    except SaveStateHookError as exc:
        logging.exception(exc)
        raise SystemExit(303_000) from exc

    except SaveCorrectionsHookError as exc:
        logging.exception(exc)
        raise SystemExit(303_000) from exc

    raise SystemExit(0)


@_project.command("task-report")
@click.argument(
    "task",
)
@click.pass_obj
def _task_report(path: Path, task: str) -> None:
    """Create short report for task.

    TASK - name pattern matching exactly one task, for which report should be created.
    """
    try:
        create_report_from(path, task)

    except AmbiguousTaskKeyError as exc:
        logging.critical(exc.args[0])
        raise SystemExit(304_00) from exc


@main.command()
@click.argument(
    "mode",
    type=click.Choice(["FSNQ", "FSNQ+", "SBS", "G3PE3Q", "G4PE3Q"]),
)
@click.option(
    "-i",
    "--input",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    help="See INPUT section.",
)
@click.option(
    "--vis",
    type=float,
    required=True,
    help="visibility against white noise. Between 0 and 1. To be used when the "
    "algorithm is stuck.",
)
@click.option(
    "--steps",
    type=int,
    required=True,
    help="Number of attempts to correct the closest separable state. A negative "
    "number adds its value to pre-existing list of results. Rounded down to a "
    "multiple of 10.",
)
@click.option(
    "--cors",
    type=int,
    required=True,
    help="Maximal number of corrections to the closest separable state. A "
    "negative number adds its value to pre-existing list of results. "
    "Rounded down to a multiple of 50",
)
@click.option(
    "-o",
    "--output",
    required=False,
    default=(Path.cwd() / "out").as_posix(),
    help="Path to directory to put output files in.",
)
@click.option(
    "-s",
    "--size",
    type=int,
    default=None,
    help="Size of system",
)
@click.option(
    "-n",
    "--sub-sys-size",
    type=int,
    default=None,
    help="Size of each subsystems",
)
@click.option(
    "-t",
    "--type",
    "data_type",
    default="complex",
    type=click.Choice(["complex", "real", "int"]),
)
def file(  # pylint: disable=too-many-arguments
    vis: float,
    steps: int,
    cors: int,
    mode: str,
    input_dir: str,
    output: Optional[str],
    size: Optional[str],
    sub_sys_size: Optional[str],
    data_type: str,
) -> None:
    """
    \b
    MODE:
        FSNQ    -   full separability of an n-quDit state.
        FSNQ+   -   full separability of an n-quDit state (d1 optional and can be
                    arbitrary).
        SBS     -   separability of a bipartite state.
        G3PE3Q  -   genuine 3-partite entanglement of a 3-quDit state
        G4PE3Q  -   genuine 4-partite entanglement of a 3-quDit state

    \b
    INPUT:
        Path to the directory containing input files, folder should be named with
        prefix used by input files inside. All files are expected to be in
        Matrix Market exchange format.
        Files which can be included:
            - {prefix}_in.mtx - the input state. (Required)
            - {prefix}_sym_0_0.mtx,{prefix}_sym_0_1.mtx,... - (Optional)
                symmetry unitaries applied to the output state. The first number
                following the prefix is the symmetry label, the second is the manifold.
            - {prefix}_proj.mtx - (Optional) projections applied to the output state.

    \b
    Output:
    -------
    If these files exist, the program will resume from the last record:
    -   prefix_mode_d1_out_vis.mtx: final separable state (can be used as an
        initial separable state)
    -   prefix_mode_d1_list_vis.mtx: number of steps, corrections, and the
        squared HS distance every 50 corrections

    -   prefix_report_mode_d1_vis.txt: The report file.
    -   prefix_abort.txt: The error message if the algorithm was extremely slow
        (for some highly entangled states).
    """

    # String formatting reference: https://peps.python.org/pep-3101/
    logging.debug("INPUT PARAMETERS")
    logging.debug("================")
    logging.debug("      vis         =   {0!r}", vis)
    logging.debug("      steps       =   {0!r}", steps)
    logging.debug("      cors        =   {0!r}", cors)
    logging.debug("      mode        =   {0!r}", mode)
    logging.debug("      input       =   {0!r}", input_dir)
    logging.debug("      output      =   {0!r}", output)
    logging.debug("      size        =   {0!r}", size)
    logging.debug("  sub_sys_size    =   {0!r}", sub_sys_size)

    raise NotImplementedError(data_type)
