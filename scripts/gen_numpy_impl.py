"""Script for generating numpy precision backends from template."""

import logging
from pathlib import Path
import click
import jinja2
import black
import black.mode


TEMPLATES_DIR = Path(__file__).parent / "templates"
DEFAULT_DEST = Path(__file__).parent.parent / "cssfinder/algorithm/backend/numpy/_impl"


@click.command()
@click.option("--dest", default=DEFAULT_DEST, type=Path)
@click.option("-d", "--debug", is_flag=True, default=False)
@click.option("--debug-dtype-checks", is_flag=True, default=False)
def main(dest: Path, debug: bool, debug_dtype_checks: bool) -> None:
    """Generate numpy precision backends from template."""

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR.as_posix()),
        autoescape=jinja2.select_autoescape(),
    )
    template = env.get_template("numpy.pyjinja2")

    for name, primary, floating, complex_ in [
        ("_complex128.py", "np.complex128", "np.float64", "np.complex128"),
        ("_complex64.py", "np.complex64", "np.float32", "np.complex64"),
        ("_float64.py", "np.float64", "np.float64", "np.complex128"),
        ("_float32.py", "np.float32", "np.float32", "np.complex64"),
    ]:
        logging.warning("Rendering %r %r %r %r", name, primary, floating, complex_)
        source = template.render(
            primary=primary,
            floating=floating,
            complex=complex_,
            is_debug=debug,
            debug_dtype_checks=debug_dtype_checks,
            is_floating=("float" in primary),
        )
        source = black.format_str(source, mode=black.mode.Mode())
        (dest / name).write_text(source, "utf-8")

    raise SystemExit(0)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
