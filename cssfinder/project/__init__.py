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


"""Contains utilities directly corrected with structure and operations of
cssfproject.json files."""


from __future__ import annotations

import json
from pathlib import Path

from cssfinder.log import get_logger
from cssfinder.project.cssfproject import CSSFProject


def load_project_from(file_or_directory: str | Path) -> CSSFProject:
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
        file_path = file_or_directory
        del file_or_directory

    else:
        file_path = file_or_directory
        del file_or_directory

    logger.debug(f"Resolved project path to {file_path.as_posix()!r}")

    content = file_path.read_text(encoding="utf-8")
    try:
        content = json.loads(content)
    except json.JSONDecodeError as exc:
        raise MalformedProjectFileError() from exc

    if not isinstance(content, dict):
        logger.critical("Content of cssfproject.json file is not a dictionary.")
        raise InvalidCSSFProjectContent(content)

    project = CSSFProject(**content)
    project.set_file_path(file_path)
    return project


class InvalidCSSFProjectContent(ValueError):
    """Raised by load_from() when file content is not a dictionary."""


class MalformedProjectFileError(ValueError):
    """Rased when project file content can't be correctly decoded."""
