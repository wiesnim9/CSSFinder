from __future__ import annotations

import logging
import logging.handlers as handlers
import sys
from pathlib import Path
from typing import Any, MutableMapping

LOGGER: logging.Logger
VERBOSITY_MAP: dict[int, int] = {
    0: logging.CRITICAL,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG,
}
LOG_ENCODING: str = "utf-8"
LOG_DIR: Path = Path("log")
LOG_FILE: Path = LOG_DIR / "cssfinder.log"


from inspect import signature


class _FormatStr:
    def __init__(self, fmt: str, args: Any, kwargs: MutableMapping[str, Any]) -> None:
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        disable_format = self.kwargs.get("extra", {}).get("disable_format", False)
        if disable_format:
            return self.fmt
        return str(self.fmt).format(*self.args, **self.kwargs)


class FormatStringAdapter(logging.LoggerAdapter):
    """Logger adapter adding .format() formatting style support to log commands."""

    def log(self, level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(level):
            msg, log_kwargs = self.process(msg, kwargs)
            self.logger._log(  # pylint: disable=protected-access
                level,
                _FormatStr(msg, args, kwargs),
                (),
                **log_kwargs,
            )

    def process(
        self, msg: _FormatStr, kwargs: MutableMapping[str, Any]
    ) -> tuple[_FormatStr, MutableMapping[str, Any]]:
        return msg, {
            key: kwargs[key]
            for key in signature(
                self.logger._log  # pylint: disable=protected-access
            ).parameters
            if key in kwargs
        }


def enable_logging(verbosity: int) -> None:
    """Configure global logger for cssfinder.

    Parameters
    ----------
    verbosity : int
        Specifies verbosity, 0 is critical, 3 or more is debug, intermediate values
        accordingly. When value exceeds this range, it is capped.
    """
    global LOGGER  # pylint: disable=global-statement

    verbosity = min(3, max(0, verbosity))

    global_verbosity = logging.DEBUG
    logger = logging.getLogger("cssfinder")
    logger.setLevel(global_verbosity)

    formatter = logging.Formatter(
        "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
        datefmt="%y.%m.%d %H:%M:%S",
    )

    sys.stderr.reconfigure(encoding=LOG_ENCODING)  # type: ignore

    stream_verbosity = VERBOSITY_MAP.get(verbosity, 3)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(stream_verbosity)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    LOG_DIR.mkdir(0o777, True, True)
    log_file_already_exists = LOG_FILE.exists()

    file_verbosity = logging.DEBUG
    file_handler = handlers.TimedRotatingFileHandler(
        LOG_FILE,
        # Only way to get full time stamp is to select "S" - seconds
        when="S",
        # Roll over during execution only after 24h, but time stamp includes seconds,
        # thus rollover on the beginning of execution will be noticed after 1 second
        interval=24 * 60 * 60,
        backupCount=10,
        encoding=LOG_ENCODING,
    )
    file_handler.setLevel(file_verbosity)
    file_handler.setFormatter(formatter)

    if log_file_already_exists:
        file_handler.doRollover()

    logger.addHandler(file_handler)

    LOGGER = FormatStringAdapter(logger)  # type: ignore


def get_logger() -> logging.Logger:
    """Return CSSFinder logger instance."""
    return LOGGER
