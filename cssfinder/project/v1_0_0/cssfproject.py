"""CSSFinder uses its own project format allowing for file based customization of
parameters used by gilbert algorithm.

This file contains implementation of project configuration in 1.0.0 version.
"""


from __future__ import annotations

from enum import Enum
from typing import ClassVar, Optional, Type, TypeVar

from packaging.version import Version
from pydantic import BaseModel, ConstrainedStr, EmailStr, Field

from cssfinder.log import get_logger
from cssfinder.project.base import CSSFProjectBase


class CSSFProjectV100(CSSFProjectBase):
    """CSSFProject file specification version 1.0.0."""

    version: ClassVar[Version] = Version("1.0.0")
    """Project file format version."""

    meta: Meta
    """Project meta information like name and author."""

    resources: Resources
    """Resources used by project, including state matrices and other files."""

    algorithm: Algorithm

    def info_display(self) -> None:
        """Display configuration content."""
        logger = get_logger()
        logger.info("\n" + self.json(indent=2), extra=dict(disable_format=True))


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

    Float64 = "Float64"
    """64bit floating point without complex values."""

    Float32 = "Float32"
    """32bit floating point without complex values."""

    Complex128 = "Complex128"
    """64 bit floating point real part with 64 bit floating point complex value."""

    Complex64 = "Complex64"
    """32 bit floating point real part with 32 bit floating point complex value."""

    # pylint: enable=invalid-name


Resources.update_forward_refs()  # type: ignore
Meta.update_forward_refs()  # type: ignore
CSSFProjectV100.update_forward_refs()  # type: ignore
