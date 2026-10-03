"""Shared tables for the documentation build (imported by gen_pages.py and hooks.py).

The site is built from ``docs/`` plus one Python API page per non-test module, generated at
build time (mkdocstrings).

Pages link to source files (``../../sim/se.py``) exactly as they would on GitHub.
:func:`rewrite_links` keeps links to other ``docs/`` pages and maps every other repository
file or directory to its GitHub source view.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
from pathlib import Path

#: Repository root (``docs/_tools/`` is two levels below it).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: GitHub base URL for source links (``mkdocs.yml`` ``repo_url``).
REPO_URL = "https://github.com/AndersonFaustino/WinHint"

#: Branch used for GitHub source links.
REPO_BRANCH = "main"

#: Python files that never get an API page (tests, test helpers, lit configuration).
API_EXCLUDE = re.compile(r"(^|/)(tests?|conftest\.py$|test_[^/]*\.py$)|^compiler/test/|^third_party/|^docs/")

#: Root of the generated Python API pages (site path).
API_ROOT = "api/python"


def tracked_files(pattern: str) -> list[str]:
    """Return repository files matching a git pathspec, tracked or untracked-but-not-ignored.

    Args:
        pattern: A git pathspec such as ``"*.py"``.

    Returns:
        Sorted repository-relative POSIX paths.
    """
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", pattern],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return sorted(p for p in out.splitlines() if (REPO_ROOT / p).is_file())


def python_api_modules() -> list[tuple[str, str]]:
    """List the Python modules that get an API page.

    Returns:
        ``(repository path, dotted identifier)`` pairs, e.g.
        ``("sim/baselines/lut/whdata.py", "sim.baselines.lut.whdata")``. Directories are
        treated as PEP 420 namespace packages rooted at the repository.
    """
    mods = []
    for path in tracked_files("*.py"):
        if API_EXCLUDE.search(path) or "." in Path(path).stem:
            continue
        mods.append((path, path[: -len(".py")].replace("/", ".")))
    return mods


#: Start or end of a fenced code block.
FENCE = re.compile(r"^(```|~~~)")
_LINK = re.compile(r"(\]\()([^)\s]+)(\))")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:|^#|^/")


def rewrite_links(markdown: str, source: str, page: str) -> str:
    """Rewrite relative Markdown links of a page whose text comes from ``source``.

    Args:
        markdown: Page text.
        source: Repository path the text was written for (links are relative to it).
        page: Site path of the page (relative to ``docs/``).

    Returns:
        The text with every relative inline link outside fenced code blocks pointing at
        the site page of the target (a ``docs/`` file) or, for any other existing
        repository file or directory, at its GitHub source view.
        Links to missing targets are left alone so that ``mkdocs build --strict`` reports
        them.
    """
    page_dir = posixpath.dirname(page)

    def fix(m: re.Match[str]) -> str:
        target = m.group(2)
        if _SCHEME.match(target):
            return m.group(0)
        path, sep, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
        if resolved.startswith("../"):
            return m.group(0)
        if resolved.startswith("docs/") and (REPO_ROOT / resolved).is_file():
            new = posixpath.relpath(resolved[len("docs/"):], page_dir or ".")
        elif (REPO_ROOT / resolved).exists():
            kind = "tree" if (REPO_ROOT / resolved).is_dir() else "blob"
            resolved = "" if resolved == "." else resolved
            new = f"{REPO_URL}/{kind}/{REPO_BRANCH}/{resolved}"
        else:
            return m.group(0)
        return f"{m.group(1)}{new}{sep}{anchor}{m.group(3)}"

    out, in_fence = [], False
    for line in markdown.splitlines(keepends=True):
        if FENCE.match(line.lstrip()):
            in_fence = not in_fence
        out.append(line if in_fence else _LINK.sub(fix, line))
    return "".join(out)
