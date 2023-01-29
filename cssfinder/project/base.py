"""This file contains metaclass and base class used to create concrete implementations
of cssfproject.json file format containers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional, Type, TypeVar, cast

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Extra, validator
from pydantic.main import ModelMetaclass

from cssfinder.log import get_logger

CSSFPROJECT_CLS_INDEX: dict[Version, Type[CSSFProjectBase]] = {}


Self = TypeVar("Self", bound="CSSFProjectMeta")


class CSSFProjectMeta(ModelMetaclass):
    """Metaclass used by all CSSFProject implementations."""

    def __new__(
        cls: type[Self],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> Self:
        cssfproject = super().__new__(cls, name, bases, namespace, **kwargs)
        CSSFPROJECT_CLS_INDEX[cssfproject.version] = cssfproject

        return cast(Self, cssfproject)


CSSFProjectBaseT = TypeVar("CSSFProjectBaseT", bound="CSSFProjectBase")


class CSSFProjectBase(BaseModel, metaclass=CSSFProjectMeta):
    """Base class used by all CSSFProject implementations."""

    version: ClassVar[Version] = Version("0.0.0")
    """Project file format version."""

    file: Optional[Path]
    """Path to loaded project file."""

    @property
    def directory(self) -> Optional[Path]:
        """Path to directory containing cssfproject.json file."""
        if self.file is None:
            return None
        return self.file

    class Config:
        validate_assignment = True
        extra = Extra.ignore

    @validator("file")
    @classmethod
    def _path_resolve(cls, value: Path) -> Path:
        if isinstance(value, Path):
            value = value.expanduser().resolve()

        return value

    @classmethod
    def new(
        cls: Type[CSSFProjectBaseT],
        data: dict[str, Any],
        project_path: Optional[Path] = None,
    ) -> CSSFProjectBaseT:
        """Create new instance of CSSFProject of selected version.

        Parameters
        ----------
        data : dict[str, Any]
            data to feed into project class.

        Returns
        -------
        CSSFProjectBase
            instance of project.

        Raises
        ------
        CSSFProjectFileMissingVersion
            When version of project is not specified.
        CSSFProjectInvalidVersion
            When version string is malformed.
        CSSFProjectVersionNotSupported
            When version is not supported.
        """
        try:
            version_str = data.pop("version")
        except KeyError as exc:
            raise CSSFProjectFileMissingVersion() from exc

        try:
            version = Version(version_str)
        except InvalidVersion as exc:
            raise CSSFProjectInvalidVersion(version_str) from exc

        if version not in CSSFPROJECT_CLS_INDEX:
            raise CSSFProjectVersionNotSupported(version)

        project_cls = CSSFPROJECT_CLS_INDEX[version]
        project = project_cls(**data, project_path=project_path)

        logger = get_logger()
        logger.debug(
            "Instantiated project class {0}.{1} from {2!r}",
            project_cls.__module__,
            project_cls.__qualname__,
            project_path.as_posix() if project_path else project_path,
        )

        if project.version < cls.version:
            raise ProjectFormatTooOld(project.version, cls.version)

        return cast(CSSFProjectBaseT, project)

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


class CSSFProjectFileMissingVersion(ValueError):
    """Raised when "version" field is missing in cssfproject.json file."""


class CSSFProjectInvalidVersion(InvalidVersion):
    """Raised when "version" field contains invalid version string."""

    def __init__(self, version: str) -> None:
        super().__init__()
        self.version = version


class CSSFProjectVersionNotSupported(InvalidVersion):
    """Raised when version requested is not supported."""

    def __init__(self, version: Version) -> None:
        super().__init__()
        self.version = version


class ProjectFormatTooOld(ValueError):
    """Raised when project format is too old to perform requested task."""

    def __init__(self, got: Version, required: Version) -> None:
        super().__init__()
        self.got = got
        self.required = required
