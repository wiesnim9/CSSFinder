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
from typing import TYPE_CHECKING

from cssfinder.algorithm.gilbert import Gilbert
from cssfinder.cssfproject import CSSFProject, GilbertCfg, Task
from cssfinder.hooks import save_corrections_hook, save_matrix_hook
from cssfinder.io.asset_loader import GilbertAssetLoader
from cssfinder.reports.manager import ReportManager

if TYPE_CHECKING:
    from pathlib import Path

    from cssfinder.reports.renderer import ReportType


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
    project.eval_dynamic()
    logging.debug("Running project %r", project.meta.name)

    message = "\n    |  ".join(project.json(indent=2).split("\n"))
    logging.info("%s", "\n    |  " + message)

    for task in project.select_tasks(tasks):
        run_task(task, is_debug=is_debug)


def run_task(task: Task, *, is_debug: bool = False) -> None:
    """Run task until completed."""
    if task.gilbert:
        run_gilbert(task.gilbert, task.output, is_debug=is_debug)


def run_gilbert(
    config: GilbertCfg,
    task_output_dir: Path,
    *,
    is_debug: bool = False,
) -> None:
    """Run Gilbert algorithm part of task."""
    asset_loader = GilbertAssetLoader()
    assets = asset_loader.load_assets(config)

    task_output_dir.mkdir(0o764, parents=True, exist_ok=True)

    algorithm = Gilbert(
        assets.state,
        mode=config.mode,
        backend=config.get_backend().name,
        precision=config.get_backend().precision,
        visibility=config.runtime.visibility,
        is_debug=is_debug,
    )
    algorithm.run(
        epochs=config.runtime.max_epochs,
        iterations=config.runtime.iters_per_epoch,
        max_corrections=config.runtime.max_corrections,
        save_state_hook=save_matrix_hook(task_output_dir / "state.mtx"),
        save_corrections_hook=save_corrections_hook(
            task_output_dir / "corrections.json",
        ),
    )


def create_report_from(
    project_file_path: Path | str, task: str, reports: list[ReportType]
) -> None:
    """Load project (`cssfproject.json`) and create report for task selected by
    pattern.
    """
    project = CSSFProject.load_project(project_file_path)
    logging.info(
        "Loaded project %r by %r <%r>.",
        project.meta.name,
        project.meta.author,
        project.meta.email,
    )
    return create_report(project, task, reports)


def create_report(project: CSSFProject, task: str, reports: list[ReportType]) -> None:
    """Create report for task selected by pattern from project object."""
    tasks = project.select_tasks([task])

    if len(tasks) > 1:
        matched_tasks_names = [t.name for t in tasks]
        message = (
            f"Pattern {task!r} matches more than one task ({len(tasks)}): "
            f"{matched_tasks_names!r}"
        )
        raise AmbiguousTaskKeyError(message)

    task_object, *_ = tasks

    manager = ReportManager(project, task_object)
    prepared_manager = manager.prepare()
    for report_type in reports:
        prepared_manager.request_report(report_type).save_to(
            task_object.output / f"report.{report_type.name.lower()}"
        )


class AmbiguousTaskKeyError(KeyError):
    """Raised during report creation when name pattern selects more than one task."""
