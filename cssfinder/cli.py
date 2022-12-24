from __future__ import annotations
import logging

from pathlib import Path
from typing import Optional

import click
import pendulum

from cssfinder.api import run
from cssfinder.io import show_logo
from cssfinder.log import enable_logging, get_logger
from cssfinder.modes import DataType
from cssfinder.task import Task


@click.command()
@click.option(
    "-v",
    "--verbose",
    default=0,
    count=True,
    help="Control verbosity of logging, by default critical only, use "
    "-v, -vv, -vvv to gradually increase it.",
)
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
def main(  # pylint: disable=too-many-arguments
    verbose: int,
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
    enable_logging(verbose)
    logger = get_logger()

    if verbose >= 2:
        show_logo()

    # String formatting reference: https://peps.python.org/pep-3101/
    logger.info("CSSFinder started at {}", pendulum.now())
    logger.debug("INPUT PARAMETERS")
    logger.debug("================")
    logger.debug("      verbose     =   {0!r}", verbose)
    logger.debug("      vis         =   {0!r}", vis)
    logger.debug("      steps       =   {0!r}", steps)
    logger.debug("      cors        =   {0!r}", cors)
    logger.debug("      mode        =   {0!r}", mode)
    logger.debug("      input       =   {0!r}", input_dir)
    logger.debug("      output      =   {0!r}", output)
    logger.debug("      size        =   {0!r}", size)
    logger.debug(" sub_sys_size    =   {0!r}", sub_sys_size)

    task = Task.new(
        mode=mode,
        visibility=vis,
        steps=steps,
        correlations=cors,
        input_dir=input_dir,
        output_dir=output,
        size=size if size is None else int(size),
        sub_sys_size=sub_sys_size if sub_sys_size is None else int(sub_sys_size),
        data_type=data_type,
    )
    run(task)
