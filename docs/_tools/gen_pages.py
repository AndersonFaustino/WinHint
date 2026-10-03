"""Generate the build-time pages of the documentation site (run by mkdocs-gen-files).

* Writes one mkdocstrings page per Python module (:func:`sitemap.python_api_modules`)
  under ``api/python/`` plus the ``SUMMARY.md`` that literate-nav turns into the
  navigation of that section.

Links are rewritten later, page by page, by ``docs/_tools/hooks.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mkdocs_gen_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sitemap  # noqa: E402

nav = mkdocs_gen_files.Nav()
with mkdocs_gen_files.open(f"{sitemap.API_ROOT}/index.md", "w") as f:
    f.write(
        "# Python API\n\n"
        "Reference for every Python module of the repository except tests, generated from\n"
        "the Google-style docstrings in the code. Modules are named after their path:\n"
        "`sim/baselines/lut/whdata.py` is `sim.baselines.lut.whdata`. Most of them are\n"
        "command-line scripts; their module docstring describes the command line.\n\n"
        "| Module | Source |\n|--------|--------|\n"
    )
    for repo_path, ident in sitemap.python_api_modules():
        rel = ident.replace(".", "/") + ".md"
        f.write(f"| [`{ident}`]({rel}) | `{repo_path}` |\n")
nav["Overview"] = "index.md"
for repo_path, ident in sitemap.python_api_modules():
    parts = ident.split(".")
    doc = f"{sitemap.API_ROOT}/{'/'.join(parts)}.md"
    nav[tuple(parts)] = "/".join(parts) + ".md"
    with mkdocs_gen_files.open(doc, "w") as f:
        f.write(f"# `{ident}`\n\n::: {ident}\n")
    mkdocs_gen_files.set_edit_path(doc, Path("..") / repo_path)

with mkdocs_gen_files.open(f"{sitemap.API_ROOT}/SUMMARY.md", "w") as f:
    f.writelines(nav.build_literate_nav())
