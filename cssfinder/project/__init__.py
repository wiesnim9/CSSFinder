from __future__ import annotations

import json
from pathlib import Path

from cssfinder.log import get_logger
from cssfinder.project.v1_0_0.cssfproject import CSSFProjectV100


def load_project_from(file_or_directory: str | Path) -> CSSFProjectV100:
    """Load CSSFinder project in at least 1.0.0 version.

    Parameters
    ----------
    file_or_directory : str | Path
        Either project file (cssfproject.json) or directory containing project file.

    Returns
    -------
    CSSFProjectV100
        Project information container.

    Raises
    ------
    InvalidCSSFProjectContent
        Raised when project file content is not a dictionary.
    """
    logger = get_logger()

    # Unify path type to Path
    file_or_directory = Path(file_or_directory).expanduser().resolve()

    # When points to directory, dir must contain cssfproject.json file
    if file_or_directory.is_dir():
        file_or_directory /= "cssfproject.json"

    logger.debug(f"Resolved project path to {file_or_directory.as_posix()!r}")

    content = file_or_directory.read_text(encoding="utf-8")
    try:
        content = json.loads(content)
    except json.JSONDecodeError as exc:
        raise MalformedProjectFileError() from exc

    if not isinstance(content, dict):
        logger.critical("Content of cssfproject.json file is not a dictionary.")
        raise InvalidCSSFProjectContent(content)

    return CSSFProjectV100.new(content, file_or_directory)


class InvalidCSSFProjectContent(ValueError):
    """Raised by load_from() when file content is not a dictionary."""


class MalformedProjectFileError(ValueError):
    """Rased when project file content can't be correctly decoded."""
