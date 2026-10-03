#!/usr/bin/env python3
"""
run_baseline.py - B0 static-window baseline (compatibility wrapper).

The old standalone ROB-size sweep has been folded into run_experiments.py:
B0 is the set of ``static_c<i>`` variants there (one per configuration of
the machine's window table, ROB/IQ/LQ/SQ scaled together, interfaces.md §3).
This wrapper keeps the old entry point working:

  python3 sim/run_baseline.py [run_experiments options]
    == python3 sim/run_experiments.py --policies static [options]

Any --policies given here is ignored.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments  # noqa: E402


def main(argv=None) -> int:
    """Run ``run_experiments.main`` with ``--policies static``.

    Any ``--policies`` option (and its values, up to the next ``--`` flag) in
    ``argv`` is removed first.

    Args:
        argv (list[str] | None): Argument list; None means ``sys.argv[1:]``.

    Returns:
        The exit status of ``run_experiments.main``.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--policies" in argv:
        i = argv.index("--policies")
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        del argv[i:j]
    return run_experiments.main(["--policies", "static", *argv])


if __name__ == "__main__":
    sys.exit(main())
