"""CSSFinder uses its own project format allowing for file based customization of
parameters used by gilbert algorithm. This file contains implementation of project
configuration in 1.0.0 version.
"""


from __future__ import annotations

from typing import ClassVar, Optional

from packaging.version import Version
from pydantic import BaseModel, ConstrainedStr, EmailStr
from cssfinder.log import get_logger

from cssfinder.project.base import CSSFProjectBase


class CSSFProjectV100(CSSFProjectBase):
    """CSSFProject file specification version 1.0.0"""

    version: ClassVar[Version] = Version("1.0.0")
    """Project file format version."""

    meta: Meta
    """Project meta information like name and author."""

    resources: Resources
    """Resources used by project, including state matrices and other files."""

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
    """Depth of system, ie. (D)imensions in qu(D)it. (d)"""

    quantity: Optional[int]
    """Quantity of systems. ie. number of qu(D)its in state. (n)"""

Resources.update_forward_refs()
Meta.update_forward_refs()
CSSFProjectV100.update_forward_refs()
