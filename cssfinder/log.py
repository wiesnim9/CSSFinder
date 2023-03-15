#!/usr/bin/python3
# Copyright 2023 Krzysztof Wiśniewski <argmaster.world@gmail.com>
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

"""Module configures the logger to log messages. It has a logger object that sends
messages to different logging handlers (file and stream), a few classes for formatters.

Constants:
----------
LOGGER: logging.Logger
    The logger object for sending messages to logging handlers.
VERBOSITY_MAP: dict
    A dictionary mapping the verbosity level to the corresponding logging level.
LOG_ENCODING: str
    The encoding type of the log file.

Functions:
----------
def configure_logger(
    *,
    verbosity: int,
    logger_name: str,
    use_rich: bool,
    log_dir: Path | None = None,
    auto_rollover_sec: int = 24 * 60 * 60,
    max_log_files: int = 10,
) -> None:
    Configures the logger and creates logging handlers for file and stream. It takes the
    following parameters:
    *   verbosity (int): The verbosity level of the logging messages. Must be an integer
        from 0 to 3. Default is 0.
    *   logger_name (str): The name of the logger. Default is an empty string.
    *   use_rich (bool): A flag to indicate whether to use the rich library to create a
        colorful log. Default is False.
    *   log_dir (Path | None): The path to the directory where log files will be
        created. Default is the current working directory.
    *   auto_rollover_sec (int): The time interval in seconds before rolling over the
        log file. Default is 86400 seconds (24 hours).
    *   max_log_files (int): The maximum number of log files to keep. Default is 10.


Classes:
--------
MarkupStripFormatter(logging.Formatter)
    A class to remove all formatting from the logging messages and return the plain
    text.

Methods
-------
    * format(record: LogRecord) -> str: Formats the record and returns the plain text.

NoHighlightRichHandler(RichHandler)
    A subclass of RichHandler to remove all rich formatting from the logging messages.

"""

from __future__ import annotations

import logging
import sys
from logging import LogRecord, getLogger, handlers
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

LOGGER: logging.Logger
VERBOSITY_MAP: dict[int, int] = {
    0: logging.CRITICAL,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG,
}
LOG_ENCODING: str = "utf-8"


__version__ = "1.0.0-rc3"


class MarkupStripFormatter(logging.Formatter):
    """A logging formatter that strips `rich` markup tags from log messages before they
    are emitted.

    Attributes
    ----------
    None

    Methods
    -------
    format(record: logging.LogRecord) -> str:
        Return a formatted and stripped version of the log message.

    """

    def format(self, record: LogRecord) -> str:  # noqa: A003
        """Return the formatted string of a log record after stripping all formatting
        tags.

        _extended_summary_

        Parameters
        ----------
        record : LogRecord
            The log record to be formatted.

        Returns
        -------
        str
            The formatted string with all formatting tags removed.

        """
        string = super().format(record)
        # render strings with rich
        seg_list = Text.from_markup(string).render(Console())
        # but use only text part to get rid of all formatting
        return "".join(seg.text for seg in seg_list)


class NoHighlightRichHandler(RichHandler):
    """RichHandler subclass which permanently disables message highlighting (coloring of
    integers, strings, etc.).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.highlighter = None  # type: ignore[assignment]


def configure_logger(
    *,
    verbosity: int,
    logger_name: str,
    use_rich: bool,
    log_dir: Path | None = None,
    auto_rollover_sec: int = 24 * 60 * 60,
    max_log_files: int = 10,
) -> None:
    """Configure a logger with a file handler and a console handler.

    Parameters
    ----------
    verbosity : int
        The verbosity level for the console handler. The value should be between 0 and
        3, with 0 being the least verbose and 3 being the most verbose.
    logger_name : str
        The name of the logger.
    use_rich : bool
        A flag indicating whether to use the rich console handler. If True, use the rich
        console handler; otherwise, use the standard console handler.
    log_dir : str or Path or None, optional
        The path to the directory where log files should be stored. If None, store the
        log files in the current working directory. The default is None.
    auto_rollover_sec : int, optional
        The number of seconds after which to rollover the log file.
        The default is 24 * 60 * 60.
    max_log_files : int, optional
        The maximum number of log files to keep. The default is 10.

    Returns
    -------
    None

    """
    # Clamp verbosity between 0 and 3.
    verbosity = min(3, max(0, verbosity))

    logger = logging.getLogger()
    # global logger must have verbosity level set to DEBUG as we want out log files
    # to contain all log messages, and using any higher level would discard debug
    # messages before passing them to file handler.
    global_verbosity = logging.DEBUG
    logger.setLevel(global_verbosity)
    logger.handlers.clear()

    # Log file handler - fixed log level to DEBUG
    file_handler = _create_file_handler(
        logger_name,
        log_dir or Path.cwd() / "log",
        auto_rollover_sec,
        max_log_files,
    )
    logger.addHandler(file_handler)

    # Console logging handler - variable log level
    stream_handler = _create_stream_handler(verbosity, use_rich=use_rich)
    logger.addHandler(stream_handler)

    matplotlib_logger = getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.WARNING)

    pandas_logger = getLogger("pandas")
    pandas_logger.setLevel(logging.WARNING)


def _create_stream_handler(verbosity: int, *, use_rich: bool) -> logging.Handler:
    # By default stderr may have different encoding which may result in coding errors
    # in least expected places in code after unfortunate use of some complex character.
    sys.stderr.reconfigure(encoding=LOG_ENCODING)  # type: ignore[attr-defined]

    if use_rich:
        # Rich library stream handler allowing for colorful log.
        stream_handler: logging.Handler = NoHighlightRichHandler(
            rich_tracebacks=False,
            markup=True,
            omit_repeated_times=False,
        )
        # Intentionally no formatter is set in this branch, rich's default is good.
    else:
        # This stream handler will not use color tagging and thanks to special formatter
        # all coloring tags compatible with rich will be removed from log.
        stream_handler = logging.StreamHandler(sys.stderr)
        formatter = MarkupStripFormatter(
            fmt="%(asctime)s [%(levelname)-5.5s] %(message)s",
            datefmt="%y.%m.%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)

    # Stream handler is only one which verbosity may vary, to avoid cluttering console
    # to much.
    stream_verbosity = VERBOSITY_MAP.get(verbosity, 3)
    stream_handler.setLevel(stream_verbosity)

    return stream_handler


def _create_file_handler(
    logger_name: str,
    log_dir: Path,
    auto_rollover_sec: int,
    max_log_files: int,
) -> logging.Handler:
    # Create logging directory before trying to put files there.
    log_dir.mkdir(0o764, parents=True, exist_ok=True)
    log_file = log_dir / logger_name
    # If file already exists, logger may decide to start appending its logs to it,
    # which is not desired - preferably, one file for one session.
    log_file_already_exists = log_file.exists()

    file_handler = handlers.TimedRotatingFileHandler(
        log_file,
        # Only way to get full time stamp is to select "S" - seconds
        when="S",
        # Roll over during execution only after 24h, but time stamp includes seconds,
        # thus rollover on the beginning of execution will be noticed after 1 second
        interval=auto_rollover_sec,
        backupCount=max_log_files,
        encoding=LOG_ENCODING,
    )
    # Log file has fixed verbosity to make it possible to determine the issue after
    # system crash.
    file_verbosity = logging.DEBUG
    file_handler.setLevel(file_verbosity)

    # Format tagging specific to rich will not make log file any easier to read,
    # therefore special formatter is used which will remove all excessive elements
    # from log.
    formatter = MarkupStripFormatter(
        fmt="%(asctime)s [%(levelname)-5.5s] %(message)s",
        datefmt="%y.%m.%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # As mentioned above, when this variable was created, we may need to manually do
    # rollover to get new file for logging after startup.
    if log_file_already_exists:
        file_handler.doRollover()

    return file_handler


if __name__ == "__main__":
    print(f"Decent log config ver. {__version__}")
    raise SystemExit(0)
