# lit configuration for the WinHint compiler tests (host, `winhint` env).
#   compiler/test/run_tests.sh            # runs these too (section 6)
#   lit -v compiler/test/lit              # standalone, from the activated env
# Substitutions:
#   %wh / %jq            build/compiler/WinHint.so / JonesIQ.so
#   %whflags / %jqflags  clang flags that load the plugin (+ default machine)
#   %machines            sim/machines/      %machine  sim/machines/riscv_ooo.json
#   %rvcc / %x86cc       clang for riscv64 (conda sysroot) / x86-64
#   %inputs              compiler/test/inputs     %root  repository root
#   %python              python3 of the env
"""lit configuration of the WinHint compiler tests.

Executed by lit (``config`` and ``lit_config`` are injected). Locates the
plugins ``WinHint.so``/``JonesIQ.so`` under ``$COMPILER_BUILD`` (default
``$WINHINT_BUILD/compiler``; aborts if missing), defines the substitutions listed
in the comment above, puts the conda LLVM tools (FileCheck, not) on ``PATH`` and
adds the features ``qemu-riscv64``, ``riscv64-gnu-objdump`` and
``x86_64-gnu-objdump`` when those tools are found. The test machine is
``$MACHINE`` (default ``sim/machines/riscv_ooo.json``); test output goes to
``$WINHINT_BUILD/compiler-tests/lit``.
"""

import os
import shutil
import sys

import lit.formats

config.name = "WinHint"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".c", ".ll", ".test"]
config.excludes = ["Inputs"]
config.test_source_root = os.path.dirname(__file__)

here = os.path.dirname(os.path.abspath(__file__))
root = os.environ.get("WINHINT_ROOT") or os.path.abspath(os.path.join(here, "..", "..", ".."))
build = os.environ.get("WINHINT_BUILD") or os.path.join(root, "build")
cbuild = os.environ.get("COMPILER_BUILD") or os.path.join(build, "compiler")
prefix = os.environ.get("CONDA_PREFIX", "")
config.test_exec_root = os.path.join(build, "compiler-tests", "lit")

wh = os.path.join(cbuild, "WinHint.so")
jq = os.path.join(cbuild, "JonesIQ.so")
machine = os.environ.get("MACHINE") or os.path.join(root, "sim", "machines", "riscv_ooo.json")
for p in (wh, jq):
    if not os.path.exists(p):
        lit_config.fatal("missing %s: run compiler/build.sh" % p)

rv = os.environ.get("RV_CFLAGS") or (
    "--target=riscv64-conda-linux-gnu --sysroot=%s/riscv64-conda-linux-gnu/sysroot "
    "--gcc-toolchain=%s -fuse-ld=lld -march=rv64gc -mabi=lp64d" % (prefix, prefix))
x86 = os.environ.get("X86_CFLAGS") or "--target=x86_64-conda-linux-gnu"
cc = "clang -Wno-unused-command-line-argument"

config.substitutions += [
    ("%whflags", "-fplugin=%s -fpass-plugin=%s -mllvm -winhint-target=%s" % (wh, wh, machine)),
    ("%jqflags", "-fplugin=%s -fpass-plugin=%s -mllvm -jones-target=%s" % (jq, jq, machine)),
    ("%wh", wh),
    ("%jq", jq),
    ("%machines", os.path.join(root, "sim", "machines")),
    ("%machine", machine),
    ("%rvcc", "%s %s" % (cc, rv)),
    ("%x86cc", "%s %s" % (cc, x86)),
    ("%inputs", os.path.join(root, "compiler", "test", "inputs")),
    ("%root", root),
    ("%python", sys.executable),
]

# FileCheck/not live in libexec/llvm in the conda LLVM packages.
path = [os.path.join(prefix, "libexec", "llvm"), os.path.join(prefix, "bin")]
config.environment["PATH"] = os.pathsep.join(path + [os.environ.get("PATH", "")])
for v in ("HOME", "CONDA_PREFIX", "WINHINT_ROOT", "WINHINT_BUILD"):
    if v in os.environ:
        config.environment[v] = os.environ[v]
config.environment["WINHINT_ROOT"] = root
config.environment["WINHINT_BUILD"] = build

if shutil.which("qemu-riscv64", path=config.environment["PATH"]):
    config.available_features.add("qemu-riscv64")
for t in ("riscv64-conda-linux-gnu-objdump", "x86_64-conda-linux-gnu-objdump"):
    if shutil.which(t, path=config.environment["PATH"]):
        config.available_features.add(t.split("-")[0] + "-gnu-objdump")
