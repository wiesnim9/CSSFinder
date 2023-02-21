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

from pathlib import Path
from typing import Optional

import click
import pendulum

import cssfinder
from cssfinder.algorithm.gilbert import Gilbert
from cssfinder.io.asset_loader import AssetLoader
from cssfinder.log import enable_logging, get_logger
from cssfinder.project import (
    InvalidCSSFProjectContent,
    MalformedProjectFileError,
    load_project_from,
)


@click.group(invoke_without_command=True, no_args_is_help=True)
@click.option(
    "-v",
    "--verbose",
    default=0,
    count=True,
    help="Control verbosity of logging, by default+ critical only, use "
    "-v, -vv, -vvv to gradually increase it.",
)
@click.version_option(cssfinder.__version__, "-V", "--version", prog_name="cssfinder")
def main(verbose: int) -> None:
    """CSSFinder is a script for finding closest separable states."""
    enable_logging(verbose)
    logger = get_logger()
    logger.info("CSSFinder started at {}", pendulum.now())

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


@main.command("project")
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--force-squash", is_flag=True, default=False)
def _project(path: str, force_squash: bool) -> None:
    """Use project file to determine runtime configuration."""
    logger = get_logger()

    try:
        project = load_project_from(path)
        logger.info(
            "Loaded project %r by %r <%r>.",
            project.meta.name,
            project.meta.author,
            project.meta.email,
        )
    except FileNotFoundError as exc:
        logger.critical("Project file not found.")
        raise SystemExit(300_000) from exc

    except MalformedProjectFileError as exc:
        logger.critical(
            "Project file content is not a valid JSON file. Fix it and try again."
        )
        raise SystemExit(301_000) from exc

    except InvalidCSSFProjectContent as exc:
        logger.critical("Project file doesn't contain valid project configuration.")
        logger.critical("Fix it and try again.")
        raise SystemExit(302_000) from exc

    project.info_display()

    asset_loader = AssetLoader(project)
    state = asset_loader.load_initial_state(force_squash)

    algorithm = Gilbert(
        state,
        mode=project.algorithm.mode,
        backend=project.algorithm.backend,
        precision=project.algorithm.precision,
        visibility=project.algorithm.visibility,
    )
    algorithm.run(
        epochs=project.algorithm.max_epochs,
        iterations=project.algorithm.iters_per_epoch,
        max_corrections=project.algorithm.max_corrections,
    )

    raise SystemExit(0)


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
    logger = get_logger()

    # String formatting reference: https://peps.python.org/pep-3101/
    logger.debug("INPUT PARAMETERS")
    logger.debug("================")
    logger.debug("      vis         =   {0!r}", vis)
    logger.debug("      steps       =   {0!r}", steps)
    logger.debug("      cors        =   {0!r}", cors)
    logger.debug("      mode        =   {0!r}", mode)
    logger.debug("      input       =   {0!r}", input_dir)
    logger.debug("      output      =   {0!r}", output)
    logger.debug("      size        =   {0!r}", size)
    logger.debug("  sub_sys_size    =   {0!r}", sub_sys_size)

    raise NotImplementedError(data_type)
