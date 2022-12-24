from pathlib import Path
from cssfinder.cli import main


FILE = Path(__file__)
DIR = FILE.parent
DATASET_DIR = DIR / "dataset_64_64"

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main(
        [
            "FSNQ",
            "-i",
            DATASET_DIR.resolve().as_posix(),
            "--vis",
            "0.4",
            "--steps",
            "2000000",
            "--cors",
            "2000",
        ]
    )
