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


"""Script for generating numpy precision backends from template."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import black
import black.mode
import click
import jinja2

TEMPLATES_DIR = Path(__file__).parent / "templates"
DEFAULT_DEST = Path(__file__).parent.parent / "cssfinder/algorithm/backend/numpy/_impl"


@click.command()
@click.option("--dest", default=DEFAULT_DEST, type=Path)
@click.option("--disable-jit", is_flag=True, default=False)
@click.option("--debug-dtype-checks", is_flag=True, default=False)
@click.option("--use-legacy-random", is_flag=True, default=False)
def main(
    dest: Path,
    *,
    disable_jit: bool,
    debug_dtype_checks: bool,
    use_legacy_random: bool,
) -> None:
    """Generate numpy precision backends from template."""
    logging.warning("Option            --disable-jit %r", disable_jit)
    logging.warning("Option     --debug-dtype-checks %r", debug_dtype_checks)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR.as_posix()),
        autoescape=jinja2.select_autoescape(),
    )
    template = env.get_template("numpy.pyjinja2")

    for name, floating, complex_ in [
        ("_complex128.py", "np.float64", "np.complex128"),
        ("_complex64.py", "np.float32", "np.complex64"),
    ]:
        logging.warning("Rendering %r %r %r", name, floating, complex_)
        source = template.render(
            floating=floating,
            complex=complex_,
            disable_jit=disable_jit,
            debug_dtype_checks=debug_dtype_checks,
            use_legacy_random=use_legacy_random,
            is_64bit="64" in floating,
            is_32bit="32" in floating,
        )
        source = black.format_str(source, mode=black.mode.Mode())
        dest_file: Path = dest / name
        dest_file.write_text(source, "utf-8")
        subprocess.run([sys.executable, "-m", "ruff", dest_file.as_posix(), "--fix"])

    raise SystemExit(0)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
