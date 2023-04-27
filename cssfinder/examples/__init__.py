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


"""Root package of examples collection."""


from __future__ import annotations

import hashlib
import shutil
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from rich.table import Table

from cssfinder.cssfproject import CSSFProject

if TYPE_CHECKING:
    from typing_extensions import Self

EXAMPLES_DIR = Path(__file__).parent


class Example(Enum):
    """Enumeration of available examples."""

    e5qubits_json = "5qubits_json"
    e5qubits_py = "5qubits_py"
    GHZ3_json = "GHZ3_json"
    GHZ4_json = "GHZ4_json"
    SBiPa_json = "SBiPa_json"
    benchmark_32x32 = "benchmark_32x32"
    benchmark_64x64 = "benchmark_64x64"

    def get_sha256(self) -> hashlib._Hash:  # noqa: SLF001
        """Calculate and return SHA-256 of example project file."""
        source = self.get_project().project_file
        content = source.read_bytes()
        return hashlib.sha256(content)

    def get_project(self) -> CSSFProject:
        """Return project object from example."""
        return CSSFProject.load_project(self.get_path())

    def get_path(self) -> Path:
        """Return path to directory containing example."""
        return EXAMPLES_DIR / self.value

    @property
    def folder_name(self) -> str:
        """Return name of project folder."""
        return self.value

    @classmethod
    def get_info_table(cls) -> Table:
        """Create rich Table object containing information about available examples."""
        table = Table(title="Available examples.", show_lines=True)
        table.add_column("Name", justify="right", no_wrap=True, style="deep_sky_blue1")
        table.add_column("SHA", justify="center", no_wrap=True)
        table.add_column("Author", justify="center", no_wrap=False)
        table.add_column("Version", justify="center", no_wrap=True)
        table.add_column("Tasks", justify="center", no_wrap=True)
        table.add_column("Description", justify="left", no_wrap=False, style="green")

        for entry in cls:
            try:
                project = entry.get_project()

                table.add_row(
                    entry.value,
                    entry.get_sha256().hexdigest()[:8],
                    project.meta.author,
                    project.meta.version,
                    f"{len(project.tasks)}",
                    project.meta.description,
                )
            except FileNotFoundError:
                table.add_row(
                    entry.value,
                    "---",
                    "Broken",
                    "---",
                    "---",
                    "---",
                )

        return table

    @classmethod
    def select_by_name(cls, name: str) -> Self:
        """Select Example by its name."""
        for example in cls:
            if example.value == name:
                return example

        msg = f"Example with name {name} not found."
        raise ExampleNotFoundError(msg)

    @classmethod
    def select_by_sha256(cls, sha: str) -> Self:
        """Select Example by its sha."""
        for example in cls:
            if example.get_sha256().hexdigest().startswith(sha):
                return example

        msg = f"Example with sha {sha!r} not found."
        raise ExampleNotFoundError(msg)

    def clone(self, dest: Path) -> None:
        """Clone project folder to different destination."""
        src = self.get_path()

        dest_dir = dest / src.name
        dest_dir.mkdir(0o777, parents=True, exist_ok=True)

        for file in src.iterdir():
            relative_path = file.relative_to(src).as_posix()

            if relative_path.startswith(("__pycache__", "output")):
                continue

            if file.is_dir():
                shutil.copytree(file.as_posix(), (dest_dir / relative_path).as_posix())
            else:
                shutil.copy(file.as_posix(), (dest_dir / relative_path).as_posix())


class ExampleNotFoundError(KeyError):
    """Raised when example is not found."""
