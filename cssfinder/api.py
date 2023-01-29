"""EOL module."""

from __future__ import annotations

import numpy as np

from cssfinder._gilbert import Gilbert
from cssfinder.io.matrix import MatrixIO
from cssfinder.log import get_logger
from cssfinder.task import Task
from cssfinder.types import MatrixT


def run(task: Task) -> None:
    """Run algorithm task.

    Parameters
    ----------
    task : Task
        Task to execute.
    """
    logger = get_logger()
    logger.debug("Received new task ID {}", id(task))
    logger.debug("{}", task.describe())

    mtx = load_matrix(task)

    if is_real_only(mtx):
        logger.warning(
            "All values in initial state matrix are real, "
            + "imaginary parts will be discarded."
        )
        mtx = np.real(mtx)

    algorithm = Gilbert(task.mode, mtx, task.size, task.sub_sys_size)
    algorithm.run(task.visibility, task.steps, task.correlations)


def load_matrix(task: Task) -> MatrixT:
    """Load matrix from task.

    Parameters
    ----------
    task : Task
        Task to load matrix for.

    Returns
    -------
    _type_
        Loaded matrix.
    """
    matrix_io = MatrixIO.new(task.get_input_file())
    mtx = matrix_io.load()
    get_logger().debug("Loaded matrix; shape {!r} dtype {}", mtx.shape, mtx.dtype)
    return mtx


def is_real_only(mtx: MatrixT) -> bool:
    """Check if matrix contains only real values. (Imaginary parts are all 0.)

    Parameters
    ----------
    mtx : MtxT
        Matrix to check.

    Returns
    -------
    bool
        True when no imaginary parts found.
    """
    # Check if any value has imaginary part:
    has_imaginary = np.imag(mtx).any()
    # Return opposite; it means True when no imaginary parts found
    return not has_imaginary
