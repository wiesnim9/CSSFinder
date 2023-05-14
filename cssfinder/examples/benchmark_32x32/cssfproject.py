# noqa: INP001
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


"""Contains example of dynamically generated configuration.

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

TASKS = {}

for backend in ["numpy_cython", "numpy", "numpy_jit", "rust_naive"]:
    for precision in [Precision.SINGLE, Precision.DOUBLE]:
        for i in range(64):
            TASKS[f"{backend}_{precision.value}_{i}"] = Task(
                gilbert=GilbertCfg(
                    mode=AlgoMode.FSnQd,
                    backend=BackendCfg(
                        name=backend,
                        precision=precision,
                    ),
                    state=State(
                        file=(Path(__file__).parent / "5qubits_in.mtx").as_posix()
                    ),
                    runtime=RuntimeCfg(
                        visibility=0.4,
                        max_epochs=1000,
                        iters_per_epoch=1000,
                        max_corrections=1000,
                    ),
                ),
            )


__project__ = CSSFProject(
    meta=Meta(
        author="Krzysztof Wiśniewski",
        email=EmailStr("argmaster.world@gmail.com"),
        name="benchmark_32x32",
        description="Performance benchmark with 64x64 matrices.",
        version=SemVerStr("1.0.0"),
    ),
    tasks=TASKS,
    project_path=__file__,
)

if __name__ == "__main__":
    run_project(__project__)
