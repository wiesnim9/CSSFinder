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


"""CSSFinder uses its own project format allowing for file based customization of
parameters used by gilbert algorithm.

This file contains implementation of project configuration in 1.0.0 version.
"""


from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ConstrainedStr, EmailStr, Extra, Field


class CSSFProject(BaseModel):
    """CSSFProject file specification."""

    meta: Meta
    """Project meta information like name and author."""

    resources: Resources
    """Resources used by project, including state matrices and other files."""

    algorithm: Algorithm

    _file: Optional[Path] = None
    """Path to loaded project file."""

    class Config:
        validate_assignment = True
        extra = Extra.ignore
        underscore_attrs_are_private = True

    def set_file_path(self, file: Path) -> None:
        """Set `cssfproject.json` file path."""
        self._file = file.expanduser().resolve()

    @property
    def file(self) -> Path:
        """Path to `cssfproject.json` file."""
        if self._file is None:
            return Path.cwd() / "cssfproject.json"
        return self._file

    @property
    def directory(self) -> Path:
        """Path to directory containing `cssfproject.json` file."""
        if self._file is None:
            return Path.cwd()
        return self._file.parent

    @property
    def output(self) -> Path:
        """Path to output directory for this project."""
        directory = self.directory / "output"
        directory.mkdir(0o764, True, True)
        return directory

    def expand_path(self, path: str) -> str:
        """Expand all special variables in path string.

        Parameters
        ----------
        path : str
            Path string to expand.
        project : CSSFProjectBase
            Project to expand path for, it will be used as source of some special
            variables.

        Returns
        -------
        str
            Expanded path.
        """
        return path.format(project=self)

    def info_display(self) -> None:
        """Display configuration content."""
        logging.info("%s", "\n" + self.json(indent=2))


class Meta(BaseModel):
    """Project meta information."""

    author: str
    """Author full name."""

    email: EmailStr
    """Author email address."""

    name: str
    """Name of the project."""

    description: str
    """Description of the project."""

    version: SemVerStr
    """Version of the project."""


class SemVerStr(ConstrainedStr):
    """Semantic versioning string regex, see https://semver.org/"""

    regex = (
        r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-]"
        + r"[0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+("
        + r"[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )


class Resources(BaseModel):
    """Project resources."""

    initial_state: State
    """Path to file containing initial state matrix."""

    symmetries: Optional[list[str]]
    """List of paths to files containing symmetry matrices."""

    projection: Optional[str]
    """Path to file containing projection matrix."""


class State(BaseModel):
    """State configuration."""

    file: str
    """Path to file containing state matrix."""

    depth: Optional[int]
    """Depth of system, ie.

    (D)imensions in qu(D)it. (d)
    """

    quantity: Optional[int]
    """Quantity of systems.

    ie. number of qu(D)its in state. (n)
    """


class Algorithm(BaseModel):
    """Algorithm configuration."""

    type: AlgoType
    """Type of algorithm to execute."""

    mode: AlgoMode
    """Algorithm mode to use."""

    backend: Backend
    """Backend to use for calculations."""

    precision: Precision
    """Precision of calculations."""

    visibility: float = Field(ge=0.0, le=1.0)
    """Visibility against white noise.

    Between 0 and 1.
    """

    max_epochs: int = Field(ge=1, le=1_000_000_000)
    """Maximal number of algorithm epochs to perform.

    If other interruption condition is met before the number of epochs, algorithm wont
    execute the rest of epochs.
    """

    iters_per_epoch: int = Field(ge=1, le=1_000_000)
    """Number of iterations per epochs.

    Between iterations no checks are performed, which may speed up calculations. However
    intermediate state of systems are not saved anywhere.
    """

    max_corrections: int
    """Maximal number of corrections to collect.

    Use -1 to disable this limit.
    """


_CIEnumT = TypeVar("_CIEnumT", bound="_CIEnum")


class _CIEnum(Enum):
    """Case insensitive enum base class."""

    @classmethod
    def _missing_(cls: Type[_CIEnumT], value: object) -> _CIEnumT:
        name = str(value)
        for member in cls:
            if member.name.casefold() == name.casefold():
                return member

        raise AttributeError(f"No enum value matches name {name!r}.")


class AlgoType(_CIEnum):
    """Type of algorithm."""

    GILBERT = "GILBERT"


class AlgoMode(_CIEnum):
    """Mode of algorithm."""

    # pylint: disable=invalid-name

    FSnQd = "FSnQd"
    """Full separability of n-quDit state."""

    SBiPa = "SBiPi"
    """Separability of a bipartite state."""

    G3PaE3qD = "G3PaE3qD"
    """Genuine 3-partite entanglement of a 3-quDit state."""

    G4PaE3qD = "G4PaE3qD"
    """Genuine 4-partite entanglement of a 3-quDit state."""

    # pylint: enable=invalid-name


class Backend(_CIEnum):
    """Backend to use for calculations."""

    # pylint: disable=invalid-name

    NumPy = "NumPy"

    # pylint: enable=invalid-name


class Precision(_CIEnum):
    """Precision of calculations performed."""

    # pylint: disable=invalid-name

    DOUBLE = "double"
    """64 bit floating point real part with 64 bit floating point complex value."""

    FULL = "full"
    """32 bit floating point real part with 32 bit floating point complex value."""

    # pylint: enable=invalid-name


Algorithm.update_forward_refs()  # type: ignore
Resources.update_forward_refs()  # type: ignore
Meta.update_forward_refs()  # type: ignore
CSSFProject.update_forward_refs()  # type: ignore
