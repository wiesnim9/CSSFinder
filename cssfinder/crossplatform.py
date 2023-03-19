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

import psutil

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
IS_WIN32 = System.Win32 == SYSTEM
IS_LINUX = System.Linux == SYSTEM
IS_MAC = System.MacOS == SYSTEM


if IS_WIN32:

    def open_file_explorer(path: Path) -> None:
        """Open file explorer application specific to platform."""
        subprocess.Popen(["explorer", str(path)])

    def open_terminal(path: Path) -> None:
        """Open terminal application specific to platform."""
        subprocess.Popen(["cmd", "/K", f"cd /D {path}"])

elif IS_MAC:

    def open_file_explorer(path: Path) -> None:
        """Open file explorer application specific to platform."""
        subprocess.Popen(["open", path])

    def open_terminal(path: Path) -> None:
        """Open terminal application specific to platform."""
        subprocess.Popen(["open", "-a", "Terminal", path])

elif IS_LINUX:

    def open_file_explorer(path: Path) -> None:
        """Open file explorer application specific to platform."""
        subprocess.Popen(["xdg-open", path])

    def open_terminal(path: Path) -> None:
        """Open terminal application specific to platform."""
        terminal = os.environ.get("TERMINAL", "x-terminal-emulator")
        subprocess.Popen([terminal, "--working-directory", str(path)])


if IS_WIN32:

    class Priority(Enum):
        """Process priority constants."""

        IDLE = psutil.IDLE_PRIORITY_CLASS
        BELOW_NORMAL = psutil.BELOW_NORMAL_PRIORITY_CLASS
        NORMAL = psutil.NORMAL_PRIORITY_CLASS
        ABOVE_NORMAL = psutil.ABOVE_NORMAL_PRIORITY_CLASS
        HIGH = psutil.HIGH_PRIORITY_CLASS
        REALTIME = psutil.REALTIME_PRIORITY_CLASS

elif IS_MAC or IS_LINUX:

    class Priority(Enum):  # type: ignore[no-redef]
        """Process priority constants."""

        IDLE = 19
        BELOW_NORMAL = 10
        NORMAL = 0
        ABOVE_NORMAL = -7
        HIGH = -15
        REALTIME = -20


if IS_WIN32:

    class IoPriority(Enum):
        """Process I/O niceness."""

        HIGH = psutil.IOPRIO_HIGH
        NORMAL = psutil.IOPRIO_NORMAL
        LOW = psutil.IOPRIO_LOW
        NONE = psutil.IOPRIO_VERYLOW

elif IS_LINUX:

    class IoPriority(Enum):  # type: ignore[no-redef]
        """Process I/O niceness."""

        HIGH = psutil.IOPRIO_CLASS_RT
        NORMAL = psutil.IOPRIO_CLASS_BE
        LOW = psutil.IOPRIO_CLASS_IDLE
        NONE = psutil.IOPRIO_CLASS_NONE

elif IS_MAC:

    class IoPriority(Enum):  # type: ignore[no-redef]
        """Process I/O niceness.

        Not available on MacOS.

        """

        HIGH = 0
        NORMAL = 1
        LOW = 2
        NONE = 3


def set_priority(pid: int, priority: Priority, io_priority: IoPriority) -> None:
    """Set process priority. Implemented for win32, linux and macOS, noop elsewhere.

    Can raise psutil.AccessDenied. io_priority is noop on macOS.

    """
    process = psutil.Process(pid)
    process.nice(priority.value)

    if IS_MAC:
        return

    if IS_LINUX:
        if io_priority == IoPriority.HIGH:
            process.ionice(io_priority.value, value=0)
            return

        if io_priority == IoPriority.NORMAL:
            process.ionice(io_priority.value, value=0)

    if IS_WIN32:
        process.ionice(io_priority.value)
