"""Machine-independent data paths.

Configs refer to datasets as `${DATASETS}/...` (e.g. `${DATASETS}/TUMTraf-Accid3nD/data/...`)
instead of a drive letter, so the same config runs on the Windows workstation and on
Linux (DGX Spark). Where the datasets are:

    CCTV_DATASETS   environment variable, if set
    otherwise       D:/datasets on Windows, ~/datasets elsewhere

Other environment variables ($HOME, ${VAR}) and ~ are expanded too.
"""

import os
import sys
from pathlib import Path
from typing import Union

DATASETS_ENV = "CCTV_DATASETS"
_TOKENS = ("${DATASETS}", "$DATASETS")


def datasets_root() -> Path:
    root = os.environ.get(DATASETS_ENV)
    if root:
        return Path(root).expanduser()
    return Path("D:/datasets") if sys.platform == "win32" else Path.home() / "datasets"


def expand(path: Union[str, Path, None]) -> str:
    """`path` with ${DATASETS}, environment variables and ~ resolved (POSIX separators kept)."""
    if path is None:
        return ""
    text = str(path)
    for token in _TOKENS:
        if token in text:
            text = text.replace(token, datasets_root().as_posix())
    return os.path.expanduser(os.path.expandvars(text))


def portable(path: Union[str, Path]) -> str:
    """Inverse of expand for writing configs: a path under the datasets root as
    ${DATASETS}/..., anything else unchanged (POSIX separators)."""
    p = Path(expand(path))
    try:
        return "${DATASETS}/" + p.resolve().relative_to(datasets_root().resolve()).as_posix()
    except (ValueError, OSError):
        return p.as_posix()
