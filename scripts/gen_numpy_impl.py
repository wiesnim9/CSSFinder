"""Script for generating numpy precision backends from template."""

from __future__ import annotations

import logging
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
    dest: Path, disable_jit: bool, debug_dtype_checks: bool, use_legacy_random: bool
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
        (dest / name).write_text(source, "utf-8")

    raise SystemExit(0)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
