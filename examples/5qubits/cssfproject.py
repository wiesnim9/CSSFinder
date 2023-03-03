"""This file contains example of dynamically generated configuration.

In fact, what this script does, it simply uses cssfinder as library to perform
calculations. This approach allows us to determine list of tasks without typical
limitations of static files.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import EmailStr

from cssfinder.api import run_project
from cssfinder.cssfproject import (
    AlgoMode,
    Backend,
    BackendCfg,
    CSSFProject,
    GilbertCfg,
    Meta,
    Precision,
    RuntimeCfg,
    SemVerStr,
    State,
    Task,
)

TASKS = [
    Task(
        gilbert=GilbertCfg(
            mode=AlgoMode.FSnQd,
            backend=BackendCfg(
                name=Backend.NumPy,
                precision=Precision.SINGLE,
            ),
            state=State(file=path.as_posix()),
            runtime=RuntimeCfg(
                visibility=0.4,
                max_epochs=1000,
                iters_per_epoch=1000,
                max_corrections=1000,
            ),
        ),
    )
    for path in Path(__file__).parent.glob("*_in.mtx")
]


run_project(
    CSSFProject(
        meta=Meta(
            author="Example",
            email=EmailStr("example@example.com"),
            name="5qubits",
            description="Project description",
            version=SemVerStr("1.0.0"),
        ),
        tasks=TASKS,
    )
)
