from cssfinder.log import get_logger
from cssfinder.task import Task


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
