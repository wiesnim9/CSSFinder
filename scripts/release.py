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

"""Create release branch and bump version of package."""


from __future__ import annotations

import re
import subprocess
from pathlib import Path

import click

SCRIPTS_DIR = Path(__file__).parent
ROOT_DIR = SCRIPTS_DIR.parent

PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
INIT_PATH = ROOT_DIR / "cssfinder" / "__init__.py"
README_PATH = ROOT_DIR / "README.md"


@click.group()
def main() -> None:
    """Release manager for releases."""


@main.command()
@click.argument("version", type=str)
def create(version: str) -> None:
    """Create release branch and change version of package.

    VERSION - valid SemVer version string, eg. 1.3.2, without 'v' prefix.

    """
    retval = subprocess.run(["git", "branch", "--show-current"], capture_output=True)
    print(retval.stdout.decode("utf-8"))
    is_dev = retval.stdout.decode("utf-8").startswith("dev")

    retval = subprocess.run(["git", "status"], capture_output=True)
    print(retval.stdout.decode("utf-8"))
    is_dirty = "Changes not staged for commit" in retval.stdout.decode("utf-8")

    if is_dirty:
        subprocess.run(["git", "add", "-A"])
        subprocess.run(["git", "stash"])

    if not is_dev:
        subprocess.run(["git", "switch", "dev"])

    subprocess.run(["git", "pull"])
    subprocess.run(["git", "switch", "-c", f"release/{version}"])
    replace_version(
        PYPROJECT_PATH,
        r"version\s*=\s*\"(.*?)\"\n",
        f'version = "{version}"\n',
    )
    replace_version(
        INIT_PATH,
        r"__version__\s*=\s*\"(.*?)\"\n",
        f'__version__ = "{version}"\n',
    )
    replace_version(
        README_PATH,
        r"cssfinder-(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?",
        f"cssfinder-{version}",
        count=0,
    )
    subprocess.run(["git", "add", "-A"])
    subprocess.run(["poetry", "run", "poe", "run-hooks"])
    subprocess.run(["git", "add", "-A"])
    subprocess.run(["git", "commit", "-m", f"Bump version to {version}"])
    subprocess.run(["git", "push", "--set-upstream", "origin", f"release/{version}"])

    if is_dirty:
        subprocess.run(["git", "stash", "pop"])


def replace_version(src: Path, regex: str, replacement: str, count: int = 1) -> None:
    """Read file content, replace version found with and write file."""
    pyproject = src.read_text("utf-8")
    pyproject = re.compile(regex).sub(replacement, pyproject, count)
    src.write_text(pyproject, "utf-8")


if __name__ == "__main__":
    main()
