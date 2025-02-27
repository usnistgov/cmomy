"""
Utility to turn a file into a parallel version.

This looks for a line like

_PARALLEL = False  # !!!PARALLEL_FALSE!!!

and inserts

_PARALLE = True  # !!!PARALLEL_TRUE!!!
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

FORMAT = "[%(name)s - %(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger("make_parallel_files")


def _write_file(path_in: Path, path_out: Path) -> None:
    parallel_line = re.compile(r"^_PARALLEL.*?=.*?False(.*)")
    with path_in.open() as f_in, path_out.open("w") as f_out:
        for line in f_in:
            match = parallel_line.match(line)
            if match:
                f_out.write(f"_PARALLEL = True  # Auto generated from {path_in.name}\n")
            else:
                f_out.write(line)


if __name__ == "__main__":
    src = Path(__file__).parent.parent.resolve() / "src" / "cmomy" / "_lib"

    names = [
        "push",
        "push_cov",
        "resample",
        "resample_cov",
        "indexed",
        "indexed_cov",
        "rolling",
        "rolling_cov",
    ]

    for name in names:
        path_in = (src / name).with_suffix(".py")
        path_out = src / f"{name}_parallel.py"

        if not path_in.exists():
            msg = f"Missing {path_in}"
            raise ValueError(msg)

        logger.info("%s to %s", path_in.name, path_out.name)
        _write_file(path_in, path_out)
