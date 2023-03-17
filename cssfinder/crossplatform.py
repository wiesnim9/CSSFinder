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


"""Contain platform specific implementations of common actions like opening programs."""
from __future__ import annotations

import os
import platform
import subprocess
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class System(Enum):
    """Enumeration of system names created to avoid typos."""

    Win32 = "Windows"
    MacOS = "Darwin"
    Linux = "Linux"
    Other = "Other"

    @classmethod
    def _missing_(cls, value: Any) -> System:  # noqa: ARG003
        return cls.Other


SYSTEM = System(platform.system())


if System.Win32 == SYSTEM:

    def open_file_explorer(path: Path) -> None:
        """Open file explorer application specific to platform."""
        subprocess.Popen(["explorer", str(path)])

    def open_terminal(path: Path) -> None:
        """Open terminal application specific to platform."""
        subprocess.Popen(["cmd", "/K", f"cd /D {path}"])

elif System.MacOS == SYSTEM:

    def open_file_explorer(path: Path) -> None:
        """Open file explorer application specific to platform."""
        subprocess.Popen(["open", path])

    def open_terminal(path: Path) -> None:
        """Open terminal application specific to platform."""
        subprocess.Popen(["open", "-a", "Terminal", path])

elif System.Linux == SYSTEM:

    def open_file_explorer(path: Path) -> None:
        """Open file explorer application specific to platform."""
        subprocess.Popen(["xdg-open", path])

    def open_terminal(path: Path) -> None:
        """Open terminal application specific to platform."""
        terminal = os.environ.get("TERMINAL", "x-terminal-emulator")
        subprocess.Popen([terminal, "--working-directory", str(path)])

elif System.Other == SYSTEM:

    def open_file_explorer(path: Path) -> None:  # noqa: ARG001
        """Open file explorer application specific to platform."""
        msg = f"Unsupported platform: {platform.system()}"
        raise ValueError(msg)

    def open_terminal(path: Path) -> None:  # noqa: ARG001
        """Open terminal application specific to platform."""
        msg = f"Unsupported platform: {platform.system()}"
        raise ValueError(msg)
