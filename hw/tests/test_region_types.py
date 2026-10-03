"""Tests of the R5 (Sondag) static region typing script on synthetic region tables."""
import json
import sys
from pathlib import Path

import pytest

R5 = Path(__file__).resolve().parents[1] / "baselines" / "r5_sondag"
sys.path.insert(0, str(R5))
import region_types as rt  # noqa: E402


def _write(tmp_path, data, name="k.regions.json"):
    """Write a JSON document and return its path.

    Args:
        tmp_path (Path): Directory to write into.
        data (object): JSON-serialisable document.
        name (str): File name.

    Returns:
        (Path): The written file.
    """
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def _main(monkeypatch, *argv):
    """Run ``region_types.main()`` with the given arguments.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.
        *argv (str): Command-line arguments.
    """
    monkeypatch.setattr(sys, "argv", ["region_types.py", *argv])
    rt.main()


def _types(path):
    """Parse a ``region_id type_id`` file.

    Args:
        path (Path): Output file.

    Returns:
        (tuple[str, dict[int, int]]): Header line and region → type mapping.
    """
    lines = path.read_text().splitlines()
    return lines[0], {int(a): int(b) for a, b in (ln.split() for ln in lines[1:])}


def test_load_regions_formats(tmp_path):
    """Check the mapping, wrapped, list and scalar-entry forms of the region table."""
    m = rt.load_regions(_write(tmp_path, {"regions": {"3": {"function": "f"}, "1": 7}}))
    assert m == {3: {"function": "f"}, 1: {"value": 7}}
    lst = rt.load_regions(_write(tmp_path, [{"id": 5, "function": "g"}, {"function": "h"}, 9]))
    assert lst == {5: {"id": 5, "function": "g"}, 1: {"function": "h"}, 2: {"value": 9}}


def test_numeric_features():
    """Check that ids, positions, bools and strings are ignored and nested dicts flattened."""
    r = {"id": 1, "line": 4, "function": "f", "vector": True, "loads": 3,
         "mix": {"fp": 2.5, "col": 9, "name": "x"}}
    assert rt.numeric_features(r) == {"loads": 3.0, "mix.fp": 2.5}


def test_kmeans_deterministic_clusters():
    """Check that two well-separated groups get two labels numbered by first appearance."""
    pts = [[0.0, 0.0], [10.0, 10.0], [0.1, 0.0], [10.0, 9.9], [0.0, 0.2]]
    assert rt.kmeans(pts, 2) == [0, 1, 0, 1, 0]
    assert rt.kmeans(pts, 2) == rt.kmeans(pts, 2)
    assert rt.kmeans(pts, 99) == [0, 1, 2, 3, 4]          # k clamped to the point count
    assert rt.kmeans([[1.0], [1.0]], 0) == [0, 0]           # k >= 1, zero variance handled


def test_main_features_strategy(tmp_path, monkeypatch):
    """Check that the default strategy clusters on the common numeric features."""
    regions = {"0": {"function": "a", "loads": 100, "fp": 1}, "1": {"function": "a", "loads": 1, "fp": 90},
               "2": {"function": "b", "loads": 98, "fp": 2, "extra": 5}}
    src = _write(tmp_path, {"regions": regions})
    out = tmp_path / "types.txt"
    _main(monkeypatch, str(src), "-o", str(out), "--k", "2")
    hdr, types = _types(out)
    assert "strategy: features(fp,loads)" in hdr and str(src) in hdr
    assert types[0] == types[2] != types[1]


def test_main_features_fall_back_to_function(tmp_path, monkeypatch, capsys):
    """Check the fallback to per-function typing when no numeric feature is shared."""
    src = _write(tmp_path, {"regions": {"0": {"function": "f", "loads": 1}, "1": {"function": "g"},
                                        "2": {"function": "f"}, "3": {}}})
    _main(monkeypatch, str(src))
    cap = capsys.readouterr()
    assert "falling back to --strategy function" in cap.err
    hdr, *rows = cap.out.splitlines()
    assert "strategy: function" in hdr
    assert rows == ["0 0", "1 1", "2 0", "3 2"]


def test_main_identity_strategy(tmp_path, monkeypatch):
    """Check that the identity strategy gives every region its own type."""
    src = _write(tmp_path, {"regions": {"4": {"loads": 1}, "7": {"loads": 2}}})
    out = tmp_path / "t.txt"
    _main(monkeypatch, str(src), "--strategy", "identity", "-o", str(out))
    hdr, types = _types(out)
    assert "strategy: identity" in hdr and types == {4: 4, 7: 7}


def test_main_rejects_unknown_strategy(tmp_path, monkeypatch):
    """Check that an unknown strategy is a usage error."""
    src = _write(tmp_path, {"regions": {}})
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, str(src), "--strategy", "random")
    assert e.value.code == 2


def test_main_empty_table(tmp_path, monkeypatch, capsys):
    """Check that an empty region table produces only the header."""
    _main(monkeypatch, str(_write(tmp_path, [])))
    assert capsys.readouterr().out.splitlines() == [
        f"# region_id type_id  (strategy: identity, source: {tmp_path / 'k.regions.json'})"]
