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

"""Module contains high level API of cssfinder."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import TYPE_CHECKING, Iterable

import psutil

from cssfinder.algorithm.gilbert import Gilbert
from cssfinder.algorithm.mode_util import ModeUtil
from cssfinder.crossplatform import IoPriority, Priority, set_priority
from cssfinder.cssfproject import CSSFProject, GilbertCfg, Task
from cssfinder.io.gilbert_io import GilbertIO
from cssfinder.reports.manager import ReportManager

if TYPE_CHECKING:
    from pathlib import Path

    from cssfinder.reports.renderer import Report, ReportType


def run_project_from(
    project_file_path: Path | str,
    tasks: list[str] | None = None,
    *,
    is_debug: bool = False,
    force_sequential: bool = False,
    max_parallel: int = -1,
) -> None:
    """Load project and run all tasks."""
    project = CSSFProject.load_project(project_file_path)
    logging.info(
        "Loaded project %r by %r <%r>.",
        project.meta.name,
        project.meta.author,
        project.meta.email,
    )
    run_project(
        project,
        tasks,
        is_debug=is_debug,
        force_sequential=force_sequential,
        max_parallel=max_parallel,
    )


def run_project(
    project: CSSFProject,
    tasks: list[str] | None = None,
    *,
    is_debug: bool = False,
    force_sequential: bool = False,
    max_parallel: int = -1,
) -> list[Task]:
    """Run all tasks defined in project."""
    logging.debug("Running project %r", project.meta.name)

    message = "\n    |  ".join(project.json(indent=2).split("\n"))
    logging.info("%s", "\n    |  " + message)

    task_list = project.select_tasks(tasks)

    if force_sequential:
        for _ in map(
            run_task,
            task_list,
            repeat(TaskOptions(is_debug=is_debug)),
        ):
            pass

    else:
        with ProcessPoolExecutor(
            max_parallel if max_parallel > 0 else None
        ) as executor:
            executor.map(
                run_task,
                task_list,
                repeat(TaskOptions(is_debug=is_debug)),
            )

    return task_list


@dataclass
class TaskOptions:
    """Container for extra task options."""

    is_debug: bool


def run_task(task: Task, options: TaskOptions) -> None:
    """Run task until completed."""
    try:
        set_priority(os.getpid(), Priority.REALTIME, IoPriority.HIGH)
    except (OSError, psutil.AccessDenied):
        logging.warning(
            "Failed to elevate process priority. It can negatively affect program "
            "performance if there are more programs running in background. "
            "To allow automated priority elevation run this program as super user. "
            "You can change priority manually for process PID %r.",
            os.getpid(),
            stack_info=False,
        )

    if task.gilbert:
        run_gilbert(task.gilbert, task.task_output_directory, is_debug=options.is_debug)


def run_gilbert(
    config: GilbertCfg,
    task_output_directory: Path,
    *,
    is_debug: bool = False,
) -> None:
    """Run Gilbert algorithm part of task."""
    asset_io = GilbertIO()

    task_output_directory.mkdir(0o777, parents=True, exist_ok=True)
    logging.debug("Created directory: %r", task_output_directory.as_posix())

    algorithm = create_gilbert(config, asset_io, is_debug=is_debug)

    logging.warning("Task %r started.", config.task_name)

    for epoch_index in algorithm.run(
        max_epochs=config.runtime.max_epochs,
        iterations_per_epoch=config.runtime.iters_per_epoch,
        max_corrections=config.runtime.max_corrections,
    ):
        if corrections_count := algorithm.get_corrections_count():
            corrections = algorithm.get_corrections()
            state = algorithm.get_state()

            logging.info(
                "Executing epoch %r / %r (%.1f%%) - corrections: %r best: %r",
                epoch_index + 1,
                config.runtime.max_epochs,
                ((epoch_index + 1) / config.runtime.max_epochs) * 100,
                corrections_count,
                corrections[-1][2],
            )
            asset_io.dump_state(state, config.output_state_file)
            asset_io.dump_corrections(corrections, config.output_corrections_file)

        else:
            logging.info(
                "Executing epoch %r / %r (%.1f%%) - no corrections.",
                epoch_index + 1,
                config.runtime.max_epochs,
                ((epoch_index + 1) / config.runtime.max_epochs) * 100,
            )

    logging.warning("Task %r finished.", config.task_name)


def create_gilbert(
    config: GilbertCfg, asset_io: GilbertIO, *, is_debug: bool
) -> Gilbert:
    """Create Gilbert object from configuration with help of specified IO.

    Parameters
    ----------
    config : GilbertCfg
        Algorithm configuration.
    asset_io : GilbertIO
        IO manager to use for loading assets.
    is_debug : bool
        Debug mode flag.

    Returns
    -------
    Gilbert
        Initialized

    """
    state_config = config.get_state()

    initial_state = asset_io.load_state(state_config.file)

    if state_config.is_predefined_dimensions():
        depth = state_config.get_depth()
        quantity = state_config.get_quantity()
        logging.info("Using fixed dimensions depth=%r quantity=%r", depth, quantity)

    else:
        dimensions = ModeUtil.new(config.mode).get_dimensions(initial_state)
        depth = dimensions.depth
        quantity = dimensions.quantity
        logging.info("Deduced dimensions depth=%r quantity=%r", depth, quantity)

    symmetries = asset_io.load_symmetries(config.get_resources().symmetries)
    if symmetries:
        logging.info("Loaded symmetries:")
        for i, row in enumerate(symmetries):
            logging.info("Row %r: %r", i, [repr(sym.shape) for sym in row])
    else:
        logging.info("No symmetries provided.")

    projection = asset_io.load_projection(config.get_resources().projection)
    if projection is not None:
        logging.info("Loaded projection: %r", projection.shape)
    else:
        logging.info("No projection provided.")

    algorithm = Gilbert(
        initial=initial_state,
        depth=depth,
        quantity=quantity,
        mode=config.mode,
        backend=config.get_backend().name,
        precision=config.get_backend().precision,
        visibility=config.runtime.visibility,
        is_debug=is_debug,
    )
    algorithm.set_symmetries(symmetries)
    if projection is not None:
        algorithm.set_projection(projection)

    return algorithm


def create_report_from(
    project_file_path: Path | str, task: str, reports: list[ReportType]
) -> Iterable[Report]:
    """Load project (`cssfproject.json`) and create report for task selected by pattern.

    Parameters
    ----------
    project_file_path : Path | str
        Path to cssfproject.json file or directory containing one.
    task : str
        Name or glob expression matching task name, expected to result in selection of
        single task.
    reports : list[ReportType]
        _description_

    Returns
    -------
    Iterable[Report]
        _description_

    Yields
    ------
    Iterator[Iterable[Report]]
        _description_

    """
    project = CSSFProject.load_project(project_file_path)
    logging.info(
        "Loaded project %r by %r <%r>.",
        project.meta.name,
        project.meta.author,
        project.meta.email,
    )
    yield from create_report(project, task, reports)


def create_report(
    project: CSSFProject, task: str, reports: list[ReportType]
) -> Iterable[Report]:
    """Create report for task selected by pattern from project object."""
    tasks = project.select_tasks([task])

    if len(tasks) > 1:
        matched_tasks_names = [t.task_name for t in tasks]
        message = (
            f"Pattern {task!r} matches more than one task ({len(tasks)}): "
            f"{matched_tasks_names!r}"
        )
        raise AmbiguousTaskKeyError(message)

    task_object, *_ = tasks

    manager = ReportManager(project, task_object)
    prepared_manager = manager.prepare()
    for report_type in reports:
        yield prepared_manager.request_report(report_type)


class AmbiguousTaskKeyError(KeyError):
    """Raised during report creation when name pattern selects more than one task."""
