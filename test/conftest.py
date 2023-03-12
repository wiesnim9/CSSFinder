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

"""Pytest configuration hooks."""

from __future__ import annotations

from typing import Iterable

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Define a new option, pytest hook."""
    parser.addoption(
        "--pdf-expect-fail",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Modify pytest configuration, pytest hook."""
    config.addinivalue_line("markers", "pdf_expect_fail: mark test as slow to run")


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    """Modify list of tests."""
    if config.getoption("--pdf-expect-fail"):
        skip_mark = pytest.mark.skip(
            reason="Running only @pytest.mark.pdf_expect_fail (--pdf-expect-fail used)."
        )
        for item in items:
            if "pdf_expect_fail" not in item.keywords:
                item.add_marker(skip_mark)

    else:
        skip_mark = pytest.mark.skip(reason="use --pdf-expect-fail to run.")
        for item in items:
            if "pdf_expect_fail" in item.keywords:
                item.add_marker(skip_mark)
