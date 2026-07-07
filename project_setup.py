"""CURSOR CREATED: Shared paths for illegal-dumping notebooks (Windows / Mac / Box sync)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _pkg_dir(candidate: Path) -> Path | None:
    """Return the bstpp package dir under `candidate` (either casing), or None."""
    for name in ("bstpp", "BSTPP"):
        if (candidate / name / "__init__.py").is_file():
            return candidate / name
    return None


def find_repo(start: Path | None = None) -> Path:
    """Repo root = folder containing the bstpp package and data/.

    Works with the fork layout (bstpp/ + data/ at the repo root) and the older
    project layout (BSTPP/ + code/ + data/). The `code/` dir is optional.
    """
    start = start or Path.cwd()
    for candidate in (start, *start.parents):
        if _pkg_dir(candidate) is not None and (candidate / "data").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find project root. Run the notebook from the repository root "
        "(the folder that contains the bstpp/ package and data/)."
    )


def setup_project(*, use_cpu_jax: bool = True) -> Path:
    """
    - chdir to repo root (so data/ and output/ paths work)
    - put BSTPP/ on sys.path for import bstpp
    - optional JAX CPU default
    Returns repo Path.
    """
    repo = find_repo()
    os.chdir(repo)

    # `import bstpp` resolves the package dir at the repo root, so the repo root
    # must be on sys.path. (Old layout: BSTPP/ is also directly importable this way.)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    if use_cpu_jax:
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    return repo


REPO = find_repo()
BSTPP_DIR = _pkg_dir(REPO) or (REPO / "bstpp")
DATA_DIR = REPO / "data"
OUTPUT_DIR = REPO / "output"
