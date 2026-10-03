"""Tests of tooling/check_docstrings.py on synthetic repositories (griffe is faked)."""
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLING))
import check_docstrings as cd  # noqa: E402

GOOD = '''"""Module doc."""


class A:
    """Class doc."""

    def m(self):
        """Method doc."""
        def inner():
            pass
        return inner


def f():
    """Function doc."""
'''

BAD = '''import os


class B:
    def m(self):
        """Doc."""


async def g():
    pass


def h():
    """Doc."""
    class Inner:
        pass
'''


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a fake repository root with one documented and one undocumented file.

    Args:
        tmp_path (Path): pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.

    Returns:
        (Path): The fake repository root (also set as ``REPO_ROOT``).
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "good.py").write_text(GOOD)
    (tmp_path / "pkg" / "bad.py").write_text(BAD)
    monkeypatch.setattr(cd, "REPO_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def no_griffe(monkeypatch):
    """Make ``import griffe`` fail.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "griffe", None)


def _fake_griffe(monkeypatch, fail=()):
    """Install a fake ``griffe`` whose docstring parser warns on docstrings containing ``BAD``.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.
        fail (tuple[str]): Module identifiers whose loading raises.

    Returns:
        (list[str]): Receives the identifiers passed to ``griffe.load``.
    """
    loaded = []

    class Doc:
        """Fake docstring object."""

        def __init__(self, text, where):
            """Store the text.

            Args:
                text (str): Docstring text.
                where (str): Location used in the warning.
            """
            self.text, self.where = text, where

        def parse(self, style):
            """Log a griffe warning when the text contains ``BAD``.

            Args:
                style (str): Parser name.
            """
            assert style == "google"
            if "BAD" in self.text:
                logging.getLogger("griffe").warning("%s: bad section", self.where)

    def obj(name, doc=None, members=(), alias=False):
        """Build a fake griffe object.

        Args:
            name (str): Object name.
            doc (str | None): Docstring text.
            members (tuple): Child objects.
            alias (bool): Whether the object is an alias.

        Returns:
            (SimpleNamespace): The object.
        """
        return types.SimpleNamespace(docstring=Doc(doc, name) if doc else None, is_alias=alias,
                                     members={m.name: m for m in members}, name=name)

    def load(ident, search_paths, docstring_parser, allow_inspection):
        """Return a fake module tree for ``ident``.

        Args:
            ident (str): Dotted module name.
            search_paths (list[str]): Search paths.
            docstring_parser (str): Parser name.
            allow_inspection (bool): Inspection flag.

        Returns:
            (SimpleNamespace): The module object.

        Raises:
            ImportError: If ``ident`` is in ``fail``.
        """
        loaded.append(ident)
        if ident in fail:
            raise ImportError("nope")
        return obj(ident, "mod", (obj("f", "BAD doc"), obj("g"),
                                  obj("alias", "BAD alias", alias=True)))

    monkeypatch.setitem(sys.modules, "griffe", types.SimpleNamespace(load=load))
    return loaded


def test_missing_docstrings(repo):
    """Check undocumented module/class/def detection and the nested-function exemption."""
    assert cd.missing_docstrings("pkg/good.py") == []
    assert cd.missing_docstrings("pkg/bad.py") == [
        "pkg/bad.py:1: module", "pkg/bad.py:4: class B", "pkg/bad.py:9: def g"]


def test_griffe_warnings_without_griffe(no_griffe):
    """Check that a missing griffe yields None."""
    assert cd.griffe_warnings(["pkg/good.py"]) is None


def test_griffe_warnings_collects(repo, monkeypatch):
    """Check warnings collection, alias skipping, dotted-stem skipping and load failures."""
    loaded = _fake_griffe(monkeypatch, fail=("pkg.broken",))
    w = cd.griffe_warnings(["pkg/good.py", "pkg/lit.cfg.py", "pkg/broken.py"])
    assert loaded == ["pkg.good", "pkg.broken"]
    assert w == ["f: bad section", "pkg/broken.py: griffe could not load the module: nope"]


def test_repo_python_files(repo, monkeypatch):
    """Check git listing filtering: excluded prefixes and deleted files are dropped."""
    (repo / "third_party").mkdir()
    listing = "pkg/good.py\nthird_party/x.py\ndocs/_tools/y.py\npkg/gone.py\npkg/bad.py\n"

    def fake_run(cmd, cwd, capture_output, text, check):
        """Fake ``git ls-files``."""
        assert cmd[:2] == ["git", "ls-files"] and cwd == repo
        return subprocess.CompletedProcess(cmd, 0, stdout=listing)

    monkeypatch.setattr(cd.subprocess, "run", fake_run)
    assert cd.repo_python_files() == ["pkg/bad.py", "pkg/good.py"]


def test_main_selected_files(repo, monkeypatch, capsys, no_griffe):
    """Check main() on explicit files: pass on good, fail on bad, griffe skip note."""
    monkeypatch.chdir(repo)
    assert cd.main(["pkg/good.py"]) == 0
    out = capsys.readouterr().out
    assert "coverage: 1 files, 0 undocumented" in out and "syntax:   skipped" in out
    assert cd.main(["pkg/good.py", "--require-griffe"]) == 1
    capsys.readouterr()
    assert cd.main(["pkg/bad.py"]) == 1
    assert "  pkg/bad.py:4: class B" in capsys.readouterr().out


def test_main_whole_repo_with_griffe(repo, monkeypatch, capsys):
    """Check main() without file arguments and with griffe warnings."""
    monkeypatch.setattr(cd, "repo_python_files", lambda: ["pkg/good.py"])
    _fake_griffe(monkeypatch)
    assert cd.main([]) == 1
    out = capsys.readouterr().out
    assert "syntax:   1 griffe warnings" in out and "  f: bad section" in out
