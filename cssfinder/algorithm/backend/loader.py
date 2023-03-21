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


"""Implementation of Gilbert algorithm backend loader."""
from __future__ import annotations

import importlib
import logging
import re
from functools import lru_cache
from importlib import metadata
from typing import TYPE_CHECKING, Any, ClassVar, Type

from rich.table import Table

from cssfinder.cssfproject import Precision

if TYPE_CHECKING:
    from typing_extensions import Self

    from cssfinder.algorithm.backend.base import BackendBase


LEN_KEY_TUPLE = 2


class Loader:
    """Backend loader class."""

    BACKEND_NAME_REGEX: ClassVar[re.Pattern] = re.compile(
        r"cssfinder(_|-)backend(_|-)[a-z0-9_\-]+", re.IGNORECASE
    )

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """Load all backends available in python environment.

        This is automatically called by constructor to load all backends. You can use it
        to refresh list of loaded backends.

        """
        self.backends: dict[tuple[str, Precision], Type[BackendBase]] = {}

        for dist in importlib.import_module("pkg_resources").working_set:
            if self.BACKEND_NAME_REGEX.match(dist.project_name) is None:
                continue

            module_name = dist.project_name.replace("-", "_")

            module = importlib.import_module(module_name)
            module_meta = metadata.metadata(module_name)

            export_backend = getattr(module, "export_backend", None)

            if export_backend is None:
                continue

            backends = export_backend()

            if not isinstance(backends, dict):
                logging.critical(
                    "Backend %r unsupported export format %r, expected <class 'dict'>.",
                    module_name,
                    type(backends),
                )
                continue

            self._extend_backend_index(module_name, module_meta, backends)

    def _extend_backend_index(
        self,
        module_name: str,
        module_meta: metadata.PackageMetadata,  # type: ignore[name-defined]
        backends: Any,
    ) -> None:
        for key, value in backends.items():
            if not isinstance(key, tuple) or len(key) != LEN_KEY_TUPLE:
                logging.critical(
                    "Backend %r -> %r unsupported key format, expected key to be "
                    "tuple[str, Precision], got %r.",
                    module_name,
                    value,
                    key,
                )
                continue

            name, precision = key

            if not isinstance(name, str):
                logging.critical(
                    "Backend %r -> %r unsupported key format, expected key to be "
                    "tuple[str, Precision], got %r.",
                    module_name,
                    value,
                    key,
                )
                continue

            if not isinstance(precision, Precision):
                logging.critical(
                    "Backend %r -> %r unsupported key format, expected key to be "
                    "tuple[str, Precision], got %r.",
                    module_name,
                    value,
                    key,
                )
                continue

            if len(getattr(value, "author", "")) == 0:
                value.author = module_meta["Author"]

            self.backends[(name.casefold(), precision)] = value

    @classmethod
    @lru_cache(maxsize=1)
    def new(cls) -> Self:
        """Get instance of Loader."""
        return cls()

    def get_backend(self, name: str, precision: Precision) -> Type[BackendBase]:
        """Query set of available backends with provided properties and return backend
        class if there is one meeting expectations.
        """
        try:
            return self.backends[(name.casefold(), precision)]
        except KeyError as exc:
            msg = (
                f"There is no backend with name={name!r} and precision="
                f"{precision.name!r} currently installed."
            )
            raise BackendNotAvailableError(msg) from exc

    def get_rich_table(self) -> Table:
        """Create rich Table object containing information about available backends."""
        table = Table(title="Available backends", show_lines=True)
        table.add_column("Name", justify="right", no_wrap=True, style="deep_sky_blue1")
        table.add_column("Precision", justify="center", no_wrap=True)
        table.add_column("Author", justify="center", no_wrap=False)
        table.add_column("Source", justify="left", no_wrap=False)
        table.add_column("Description", justify="left", no_wrap=False)

        for key, value in self.backends.items():
            try:
                (name, precision), cls = key, value
                table.add_row(
                    name,
                    precision.name,
                    getattr(cls, "author", ""),
                    f"{cls.__module__}.{cls.__qualname__}",
                    getattr(cls, "description", ""),
                )
            except (TypeError, ValueError):
                logging.warning("Failed to display information about backed %r", value)

        return table


class BackendNotAvailableError(KeyError):
    """Raised when backend with specified features can not be found."""
