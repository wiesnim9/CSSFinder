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
from typing import TYPE_CHECKING, Iterable

from cssfinder.algorithm.gilbert import Gilbert
from cssfinder.algorithm.mode_util import ModeUtil
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
) -> None:
    """Load project and run all tasks."""
    project = CSSFProject.load_project(project_file_path)
    logging.info(
        "Loaded project %r by %r <%r>.",
        project.meta.name,
        project.meta.author,
        project.meta.email,
    )
    run_project(project, tasks, is_debug=is_debug)


def run_project(
    project: CSSFProject,
    tasks: list[str] | None = None,
    *,
    is_debug: bool = False,
) -> None:
    """Run all tasks defined in project."""
    logging.debug("Running project %r", project.meta.name)

    message = "\n    |  ".join(project.json(indent=2).split("\n"))
    logging.info("%s", "\n    |  " + message)

    for task in project.select_tasks(tasks):
        run_task(task, is_debug=is_debug)


def run_task(task: Task, *, is_debug: bool = False) -> None:
    """Run task until completed."""
    if task.gilbert:
        run_gilbert(task.gilbert, task.task_output_directory, is_debug=is_debug)


def run_gilbert(
    config: GilbertCfg,
    task_output_directory: Path,
    *,
    is_debug: bool = False,
) -> None:
    """Run Gilbert algorithm part of task."""
    asset_io = GilbertIO()

    task_output_directory.mkdir(0o764, parents=True, exist_ok=True)

    algorithm = create_gilbert(config, asset_io, is_debug=is_debug)

    for epoch_index in algorithm.run(
        max_epochs=config.runtime.max_epochs,
        iterations_per_epoch=config.runtime.iters_per_epoch,
        max_corrections=config.runtime.max_corrections,
    ):
        logging.info(
            "Executing epoch %r / %r (%.1f%%) - corrections: %r best: %r",
            epoch_index + 1,
            config.runtime.max_epochs,
            ((epoch_index + 1) / config.runtime.max_epochs) * 100,
            algorithm.get_corrections_count(),
            algorithm.get_corrections()[-1][2]
            if algorithm.get_corrections_count() > 0
            else None,
        )
        asset_io.dump_state(algorithm.get_state(), config.output_state_file)
        asset_io.dump_corrections(
            algorithm.get_corrections(), config.output_corrections_file
        )


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

    else:
        dimensions = ModeUtil.new(config.mode).get_dimensions(initial_state)
        depth = dimensions.depth
        quantity = dimensions.quantity

    symmetries = asset_io.load_symmetries(config.get_resources().symmetries)
    projection = asset_io.load_projection(config.get_resources().projection)

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
