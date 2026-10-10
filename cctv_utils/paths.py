"""Machine-independent data paths.

Configs refer to datasets as `${DATASETS}/...` (e.g. `${DATASETS}/TUMTraf-Accid3nD/data/...`)
instead of a drive letter, so the same config runs on the Windows workstation and on
Linux (DGX Spark). Where the datasets are:

    CCTV_DATASETS   environment variable, if set
    otherwise       D:/datasets on Windows, ~/datasets elsewhere

${PROJECT} is this project's directory. Other environment variables ($HOME, ${VAR})
and ~ are expanded too. `portable` writes paths back in this form, so keys derived
from paths (the lane-marking cache) are the same on every machine.
"""

import os
import sys
from pathlib import Path
from typing import Union

DATASETS_ENV = "CCTV_DATASETS"
_TOKENS = ("${DATASETS}", "$DATASETS")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    text = text.replace("${PROJECT}", PROJECT_ROOT.as_posix())
    return os.path.expanduser(os.path.expandvars(text))


def portable(path: Union[str, Path]) -> str:
    """Inverse of expand: a path under the datasets root as ${DATASETS}/..., under the
    project as ${PROJECT}/... (relative paths are project-relative), anything else
    unchanged (POSIX separators)."""
    p = Path(expand(path))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    for base, token in ((datasets_root(), "${DATASETS}"), (PROJECT_ROOT, "${PROJECT}")):
        try:
            return f"{token}/" + p.resolve().relative_to(Path(base).resolve()).as_posix()
        except (ValueError, OSError):
            continue
    return p.as_posix()
