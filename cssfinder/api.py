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

"""This module contains high level API of cssfinder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from cssfinder.algorithm.gilbert import Gilbert
from cssfinder.cssfproject import CSSFProject, GilbertCfg, Task
from cssfinder.hooks import save_corrections_hook, save_matrix_hook
from cssfinder.io.asset_loader import GilbertAssetLoader


def run_project_file(
    project_file_path: Path | str, tasks: Optional[list[str]] = None
) -> None:
    """Load project and run all tasks."""
    project = CSSFProject.load_project(project_file_path)
    logging.info(
        "Loaded project %r by %r <%r>.",
        project.meta.name,
        project.meta.author,
        project.meta.email,
    )
    run_project(project, tasks)


def run_project(project: CSSFProject, tasks: Optional[list[str]] = None) -> None:
    """Run all tasks defined in project."""
    project.eval_dynamic()
    logging.debug("Running project %r", project.meta.name)

    message = "\n    |  ".join(project.json(indent=2).split("\n"))
    logging.info("%s", "\n    |  " + message)

    for name, task in project.tasks.items():
        if tasks is None or name in tasks:
            run_task(task, project.output / name)


def run_task(task: Task, task_output_dir: Path) -> None:
    """Run task until completed."""
    if task.gilbert:
        run_gilbert(task.gilbert, task_output_dir)


def run_gilbert(config: GilbertCfg, task_output_dir: Path) -> None:
    """Run Gilbert algorithm part of task."""
    asset_loader = GilbertAssetLoader()
    assets = asset_loader.load_assets(config)

    task_output_dir.mkdir(0o764, parents=True, exist_ok=True)

    algorithm = Gilbert(
        assets.state,
        mode=config.mode,
        backend=config.backend.name,
        precision=config.backend.precision,
        visibility=config.runtime.visibility,
    )
    algorithm.run(
        epochs=config.runtime.max_epochs,
        iterations=config.runtime.iters_per_epoch,
        max_corrections=config.runtime.max_corrections,
        save_state_hook=save_matrix_hook(task_output_dir / "state.mtx"),
        save_corrections_hook=save_corrections_hook(
            task_output_dir / "corrections.json"
        ),
    )
