# =============================================================================
# benchmarks/Makefile — build the WinHint benchmark variants (host only, no Docker)
#
# Run inside the `winhint` micromamba env (docs/interfaces.md §1), e.g.
#     ~/.local/bin/micromamba run -n winhint make -C benchmarks ARCH=riscv VARIANT=winhint
#
# Variables
#   ARCH      riscv (default; static rv64gc for gem5/QEMU) | x86 (native x86-64)
#   VARIANT   plain | winhint | oracle | oracle_hinted | jones | jones_full | pgo |
#             clairvoyance | winhint_clairvoyance            (docs/interfaces.md §5)
#             winhint_call | oracle_call  (runtime-call mode, linked with libwinhint;
#                                          for hw/, ARCH=x86 only)
#   OPT       O2 (default) | O3          (also accepted: -O2 / -O3)
#   UNROLL    on (default) | off         (off = -fno-unroll-loops)
#   TILED     off (default) | on         (-DTILED register/cache-tiled GEMM)
#   KERNELS   subset to build (default: every benchmarks/*.c with a main())
#   EMIT      asm (default) | call | none (-winhint-emit / -jones-emit; the *_call
#             variants force call)
#   MACHINE   machine JSON for the cost model (default sim/machines/riscv_ooo.json).
#             For ARCH=riscv, model-dependent variants (winhint, winhint_clairvoyance,
#             jones, jones_full, oracle_hinted, pgo; MODEL_VARIANTS) built for a
#             non-default machine go to <variant>/<machine stem>/
#             (sim/run_experiments.py looks there first);
#             `make machines` builds VARIANT for every sim/machines/*.json.
#   MIGRATION_US  P/E switch cost (µs) for the *_call builds (-winhint-switch-model=pe);
#             default: switch_cost_us from results/hw/system/migration_cost_smt-on.json
#   SWITCH_COST  cycles; empty = model default; see -winhint-switch-cost
#   ORACLE_DIR / PGO_DIR  per-kernel region maps for oracle_hinted / pgo
#             (default machine: results/oracle, results/pgo; machine M:
#             results/oracle/M/large, results/oracle/M/small). oracle_hinted and pgo
#             are per-machine builds like winhint (see MACHINE).
#   JONES_ENC setwin (default, in the contract) | setiq (proposed IQ-only tag)
#   CV_TYPE / CV_UNROLL / CV_INDIR / CV_BRANCH_PROB   Clairvoyance (B8) knobs, defaults
#             consv / 2 / 1 / 0.9; CV_TYPE in consv|specsafe|spec|multispecsafe|multispec.
#             Ranges: compiler/baselines/clairvoyance/knobs.json. Non-default values
#             add "-cv_<type>-u<unroll>-i<indir>[-bp<p>]" to the output directory.
#
# Inputs: every kernel takes argv[1] = small|large (default large; docs/interfaces.md §5).
#   Model shapes (BERT-base / GPT-2-small: H=768, FFN 3072, 12 heads) are the same for
#   both; `small` shrinks only the input (sequence/prompt/batch) so it stays simulable
#   in gem5 O3: ~40-550 M RISC-V instructions at -O2 (QEMU count), vs ~0.2-13.6 G for
#   `large`. Size macros (SEQ_SMALL, PROMPT_SMALL, ...) can be overridden with
#   CFLAGS_EXTRA=-D... . Each kernel prints one checksum/hash line; it is bit-identical
#   across QEMU, gem5 and native x86 for the same binary (FP contraction off).
#
# Output: $(WINHINT_BUILD)/benchmarks/<arch>/<variant>[<cfg>][/<machine>]/<kernel>, where <cfg> is
# empty for the default build (O2, unroll on, untiled) and otherwise e.g.
# "-O3-nounroll-tiled". Compiler side files (<kernel>.regions.json,
# <kernel>.winhint.json, <kernel>.jones*.json) are written next to the binaries.
# Nothing is written under the source tree.
#
# Targets: all (default) | <variant> (= VARIANT=<variant> all) | matrix | matrix-all |
#          machines | check | plugins | clairvoyance-passes | list | outdir | clean
#   check: builds VARIANT and plain, runs both on CHECK_INPUT (default small;
#          qemu-riscv64 for riscv) and compares stdout bit for bit.
# =============================================================================

BENCH_DIR := $(patsubst %/,%,$(dir $(realpath $(firstword $(MAKEFILE_LIST)))))
WINHINT_ROOT  ?= $(realpath $(BENCH_DIR)/..)
WINHINT_BUILD ?= $(WINHINT_ROOT)/build

ARCH      ?= riscv
VARIANT   ?= plain
OPT       ?= O2
override OPT := $(patsubst -%,%,$(OPT))
UNROLL    ?= on
TILED     ?= off
EMIT      ?= asm
JONES_ENC ?= setwin
SWITCH_COST ?=

OUT_ROOT       ?= $(WINHINT_BUILD)/benchmarks
COMPILER_BUILD ?= $(WINHINT_BUILD)/compiler
LIBWINHINT_DIR ?= $(WINHINT_BUILD)/libwinhint
MACHINE        ?= $(WINHINT_ROOT)/sim/machines/riscv_ooo.json
RESULTS_DIR    ?= $(WINHINT_ROOT)/results
DEFAULT_MACHINE := $(WINHINT_ROOT)/sim/machines/riscv_ooo.json
MACHINE_NAME := $(basename $(notdir $(MACHINE)))
# Per-region maps (sim/baselines/oracle/oracle_sweep.py layout). The default machine
# uses the copies named in docs/interfaces.md §5; any other machine M reads
# results/oracle/M/large (oracle_hinted) and results/oracle/M/small (pgo, B7 profile).
ifeq ($(abspath $(MACHINE)),$(abspath $(DEFAULT_MACHINE)))
  ORACLE_DIR   ?= $(RESULTS_DIR)/oracle
  PGO_DIR      ?= $(RESULTS_DIR)/pgo
else
  ORACLE_DIR   ?= $(RESULTS_DIR)/oracle/$(MACHINE_NAME)/large
  PGO_DIR      ?= $(RESULTS_DIR)/oracle/$(MACHINE_NAME)/small
endif
CV_SCRIPT      ?= $(WINHINT_ROOT)/compiler/baselines/clairvoyance/cv_compile.sh

WINHINT_SO := $(COMPILER_BUILD)/WinHint.so
JONES_SO   := $(COMPILER_BUILD)/JonesIQ.so

# Kernels = every C file with a main().
KERNELS ?= $(sort $(basename $(notdir $(shell grep -l '^int main' $(BENCH_DIR)/*.c))))

VARIANTS := plain winhint oracle oracle_hinted jones jones_full pgo clairvoyance \
            winhint_clairvoyance winhint_call oracle_call
ifeq ($(filter $(VARIANT),$(VARIANTS)),)
  $(error unknown VARIANT=$(VARIANT); expected one of: $(VARIANTS))
endif
CALL_VARIANTS := winhint_call oracle_call
ifneq ($(filter $(VARIANT),$(CALL_VARIANTS)),)
  override EMIT := call
endif

# ── Configuration suffix ─────────────────────────────────────────────────────
CFG_SUFFIX :=
ifneq ($(OPT),O2)
  CFG_SUFFIX := $(CFG_SUFFIX)-$(OPT)
endif
ifeq ($(UNROLL),off)
  CFG_SUFFIX := $(CFG_SUFFIX)-nounroll
endif
ifeq ($(TILED),on)
  CFG_SUFFIX := $(CFG_SUFFIX)-tiled
endif
# B8 knobs (validated again by cv_compile.sh); non-default settings get their own dir.
CV_TYPE        ?= consv
CV_UNROLL      ?= 2
CV_INDIR       ?= 1
CV_BRANCH_PROB ?= 0.9
ifneq ($(filter clairvoyance winhint_clairvoyance,$(VARIANT)),)
  ifeq ($(filter $(CV_TYPE),consv specsafe spec multispecsafe multispec),)
    $(error CV_TYPE=$(CV_TYPE); expected consv|specsafe|spec|multispecsafe|multispec)
  endif
  CV_CFG := $(CV_TYPE)-u$(CV_UNROLL)-i$(CV_INDIR)$(if $(filter-out 0.9,$(CV_BRANCH_PROB)),-bp$(CV_BRANCH_PROB))
  ifneq ($(CV_CFG),consv-u2-i1)
    CFG_SUFFIX := $(CFG_SUFFIX)-cv_$(CV_CFG)
  endif
endif
MODEL_VARIANTS := winhint winhint_clairvoyance jones jones_full oracle_hinted pgo
MACHINE_SUBDIR :=
ifeq ($(ARCH),riscv)
  ifneq ($(filter $(VARIANT),$(MODEL_VARIANTS)),)
    ifneq ($(abspath $(MACHINE)),$(abspath $(DEFAULT_MACHINE)))
      MACHINE_SUBDIR := /$(MACHINE_NAME)
    endif
  endif
endif
OUT_DIR := $(OUT_ROOT)/$(ARCH)/$(VARIANT)$(CFG_SUFFIX)$(MACHINE_SUBDIR)

# ── Toolchain (conda env: LLVM clang + conda sysroots, see build/ENV.md) ────
# The env exports CC=<conda gcc>; the benchmarks are built with clang (LLVM plugins),
# so the compiler variable is CLANG, not CC.
CLANG ?= clang
CFLAGS_EXTRA ?=
CFLAGS_COMMON := -$(OPT) -std=c11 -Wall -gline-tables-only $(CFLAGS_EXTRA)
ifeq ($(UNROLL),off)
  CFLAGS_COMMON += -fno-unroll-loops
endif
ifeq ($(TILED),on)
  CFLAGS_COMMON += -DTILED
endif

ifeq ($(ARCH),riscv)
  # clang --target with the conda RISC-V sysroot and the cross GCC's crt/libgcc.
  RISCV_TRIPLE  ?= riscv64-conda-linux-gnu
  RISCV_SYSROOT ?= $(CONDA_PREFIX)/$(RISCV_TRIPLE)/sysroot
  TARGET_FLAGS := --target=$(RISCV_TRIPLE) --sysroot=$(RISCV_SYSROOT) \
                  --gcc-toolchain=$(CONDA_PREFIX) -march=rv64gc -mabi=lp64d
  LDFLAGS_ARCH := -static -fuse-ld=lld
  RUN ?= qemu-riscv64
  CV_ARCH := riscv64
else ifeq ($(ARCH),x86)
  X86_MARCH ?= x86-64-v3
  X86_TRIPLE ?= x86_64-conda-linux-gnu
  X86_SYSROOT ?= $(CONDA_PREFIX)/$(X86_TRIPLE)/sysroot
  TARGET_FLAGS := --target=$(X86_TRIPLE) --sysroot=$(X86_SYSROOT) \
                  --gcc-toolchain=$(CONDA_PREFIX) -march=$(X86_MARCH)
  LDFLAGS_ARCH := -fuse-ld=lld
  RUN ?=
  CV_ARCH := x86_64
else
  $(error ARCH must be riscv or x86)
endif

LDLIBS := -lm
CALL_FLAGS :=
ifeq ($(EMIT),call)
  # P/E migration cost measured by hw/ (bench_migration), if available.
  MIGRATION_JSON ?= $(RESULTS_DIR)/hw/system/migration_cost_smt-on.json
  MIGRATION_US ?= $(shell python3 -c "import json;print(json.load(open('$(MIGRATION_JSON)'))['switch_cost_us'])" 2>/dev/null)
  CALL_FLAGS := -I$(WINHINT_ROOT)/hw/libwinhint -mllvm -winhint-switch-model=pe \
                $(if $(MIGRATION_US),-mllvm -winhint-migration-us=$(MIGRATION_US))
  ifeq ($(ARCH),riscv)
    $(error EMIT=call / $(VARIANT): libwinhint is built for x86 only (hw/); use ARCH=x86)
  endif
  LDLIBS := -L$(LIBWINHINT_DIR) -l:libwinhint.a $(LDLIBS)
endif

# ── Pass flags per variant ───────────────────────────────────────────────────
WH_LOAD   = -fplugin=$(WINHINT_SO) -fpass-plugin=$(WINHINT_SO)
WH_COMMON = $(WH_LOAD) -mllvm -winhint-target=$(MACHINE) -mllvm -winhint-emit=$(EMIT) \
            -mllvm -winhint-out-dir=$(OUT_DIR) -mllvm -winhint-kernel=$(1) \
            $(if $(SWITCH_COST),-mllvm -winhint-switch-cost=$(SWITCH_COST))
JONES_COMMON = -fplugin=$(JONES_SO) -fpass-plugin=$(JONES_SO) \
            -mllvm -jones-target=$(MACHINE) -mllvm -jones-emit=$(EMIT) \
            -mllvm -jones-out-dir=$(OUT_DIR) -mllvm -jones-kernel=$(1)

pass_flags_plain                =
pass_flags_winhint              = $(call WH_COMMON,$(1)) -mllvm -winhint-mode=setwin
pass_flags_winhint_call         = $(call WH_COMMON,$(1)) -mllvm -winhint-mode=setwin $(CALL_FLAGS)
pass_flags_oracle               = -DWINHINT_ORACLE $(call WH_COMMON,$(1)) -mllvm -winhint-mode=regions
pass_flags_oracle_call          = -DWINHINT_ORACLE $(call WH_COMMON,$(1)) -mllvm -winhint-mode=regions $(CALL_FLAGS)
pass_flags_oracle_hinted        = $(call WH_COMMON,$(1)) -mllvm -winhint-mode=from-json=$(ORACLE_DIR)/$(1).json
pass_flags_pgo                  = $(call WH_COMMON,$(1)) -mllvm -winhint-mode=from-json=$(PGO_DIR)/$(1).json
pass_flags_jones                = $(call JONES_COMMON,$(1)) $(if $(filter call,$(EMIT)),-I$(WINHINT_ROOT)/hw/libwinhint) -mllvm -jones-mode=iq -mllvm -jones-iq-encoding=$(JONES_ENC)
pass_flags_jones_full           = $(call JONES_COMMON,$(1)) -mllvm -jones-mode=full
pass_flags_clairvoyance         =
pass_flags_winhint_clairvoyance = $(call WH_COMMON,$(1)) -mllvm -winhint-mode=setwin

PLUGIN_DEP :=
ifneq ($(filter winhint winhint_call oracle oracle_call oracle_hinted pgo winhint_clairvoyance,$(VARIANT)),)
  PLUGIN_DEP := $(WINHINT_SO)
endif
ifneq ($(filter jones jones_full,$(VARIANT)),)
  PLUGIN_DEP := $(JONES_SO)
endif

MAP_DEP :=
ifeq ($(VARIANT),oracle_hinted)
  MAP_DEP := $(ORACLE_DIR)/%.json
endif
ifeq ($(VARIANT),pgo)
  MAP_DEP := $(PGO_DIR)/%.json
endif

BINS := $(addprefix $(OUT_DIR)/,$(KERNELS))

.PHONY: all matrix matrix-all machines check plugins clairvoyance-passes list outdir clean guard $(VARIANTS)
.DEFAULT_GOAL := all

all: guard $(BINS)
	@echo "[OK] $(ARCH)/$(VARIANT)$(CFG_SUFFIX): $(words $(BINS)) kernels in $(OUT_DIR)"

guard:
	@case "$(abspath $(OUT_ROOT))/" in "$(BENCH_DIR)/"*) echo "[ERROR] OUT_ROOT must not be under $(BENCH_DIR)"; exit 1;; esac
	@command -v $(CLANG) >/dev/null || { echo "[ERROR] $(CLANG) not found: activate the winhint env (docs/interfaces.md §1)"; exit 1; }

$(OUT_DIR):
	mkdir -p $@

ifneq ($(filter clairvoyance winhint_clairvoyance,$(VARIANT)),)
# B8: Clairvoyance passes transform the IR; LLVM 23 lowers it.
$(OUT_DIR)/%: $(BENCH_DIR)/%.c $(PLUGIN_DEP) $(MAP_DEP) | $(OUT_DIR)
	CV_ARCH=$(CV_ARCH) CLANG="$(CLANG)" WINHINT_BUILD="$(WINHINT_BUILD)" \
	  CV_TYPE=$(CV_TYPE) CV_UNROLL=$(CV_UNROLL) CV_INDIR=$(CV_INDIR) CV_BRANCH_PROB=$(CV_BRANCH_PROB) \
	  CFLAGS="$(CFLAGS_COMMON) $(TARGET_FLAGS)" LDFLAGS="$(LDFLAGS_ARCH)" LDLIBS="$(LDLIBS)" \
	  PASS_FLAGS="$(call pass_flags_$(VARIANT),$*)" \
	  bash $(CV_SCRIPT) $< $@
else
$(OUT_DIR)/%: $(BENCH_DIR)/%.c $(PLUGIN_DEP) $(MAP_DEP) | $(OUT_DIR)
	$(CLANG) $(CFLAGS_COMMON) $(TARGET_FLAGS) $(call pass_flags_$(VARIANT),$*) $< -o $@ $(LDFLAGS_ARCH) $(LDLIBS)
endif

# Existence checks only (test -f keeps `make -B` from failing on files that exist).
$(WINHINT_SO) $(JONES_SO):
	@test -f $@ || { echo "[ERROR] $@ missing: build the LLVM plugins first: tooling/winhint.sh compiler:build (or 'make plugins')"; exit 1; }

$(ORACLE_DIR)/%.json:
	@test -f $@ || { echo "[ERROR] $@ missing: VARIANT=oracle_hinted needs the B1 per-region map for machine $(MACHINE_NAME) (sim/baselines/oracle/oracle_sweep.py; gem5 RISCV_winhint), or set ORACLE_DIR"; exit 1; }
$(PGO_DIR)/%.json:
	@test -f $@ || { echo "[ERROR] $@ missing: VARIANT=pgo needs the B7 profile of the small input for machine $(MACHINE_NAME) (compiler/baselines/pgo/pgo_flow.py all --kernel $*, or oracle_sweep.py --input small), or set PGO_DIR"; exit 1; }

plugins:
	bash $(WINHINT_ROOT)/tooling/winhint.sh compiler:build

# B8 passes (LLVM 3.8, env winhint-llvm38) -> $(WINHINT_BUILD)/clairvoyance/lib
clairvoyance-passes:
	bash $(WINHINT_ROOT)/tooling/winhint.sh clairvoyance:build

# Variants that need no profile/oracle map and no LLVM 3.8, built one after the other.
MATRIX_VARIANTS ?= plain winhint oracle jones jones_full
matrix:
	@for v in $(MATRIX_VARIANTS); do \
	  $(MAKE) --no-print-directory ARCH=$(ARCH) OPT=$(OPT) UNROLL=$(UNROLL) TILED=$(TILED) \
	    KERNELS="$(KERNELS)" VARIANT=$$v || exit 1; \
	done

# Full build matrix: MATRIX_VARIANTS x OPT {O2,O3} x UNROLL {on,off} x TILED {off,on}.
MATRIX_CONFIGS ?= O2:on:off O2:off:off O2:on:on O2:off:on O3:on:off O3:off:off O3:on:on O3:off:on
matrix-all:
	@for c in $(MATRIX_CONFIGS); do \
	  o=$${c%%:*}; r=$${c#*:}; u=$${r%%:*}; t=$${r#*:}; \
	  $(MAKE) --no-print-directory ARCH=$(ARCH) OPT=$$o UNROLL=$$u TILED=$$t \
	    KERNELS="$(KERNELS)" matrix || exit 1; \
	done

# Shorthands: `make clairvoyance`, `make ARCH=x86 winhint_call oracle_call`, ...
# (= make VARIANT=<name>; several names are built one after the other).
$(VARIANTS):
	@$(MAKE) --no-print-directory ARCH=$(ARCH) OPT=$(OPT) UNROLL=$(UNROLL) TILED=$(TILED) \
	  KERNELS="$(KERNELS)" MACHINE=$(MACHINE) VARIANT=$@ all

# VARIANT for every machine description (portability study).
MACHINES ?= $(wildcard $(WINHINT_ROOT)/sim/machines/*.json)
machines:
	@for m in $(MACHINES); do \
	  $(MAKE) --no-print-directory ARCH=$(ARCH) OPT=$(OPT) UNROLL=$(UNROLL) TILED=$(TILED) \
	    KERNELS="$(KERNELS)" VARIANT=$(VARIANT) MACHINE=$$m || exit 1; \
	done

# Bit-identical stdout of VARIANT vs plain (same ARCH/cfg) on CHECK_INPUT.
# Call-mode binaries run with WINHINT_MODE=log (decide, never migrate).
CHECK_INPUT ?= small
check: all
	@$(MAKE) --no-print-directory ARCH=$(ARCH) OPT=$(OPT) UNROLL=$(UNROLL) TILED=$(TILED) \
	  KERNELS="$(KERNELS)" VARIANT=plain
	@fail=0; for k in $(KERNELS); do \
	  ref=$$($(RUN) $(OUT_ROOT)/$(ARCH)/plain$(CFG_SUFFIX)/$$k $(CHECK_INPUT) 2>/dev/null); rr=$$?; \
	  out=$$(WINHINT_MODE=log $(RUN) $(OUT_DIR)/$$k $(CHECK_INPUT) 2>/dev/null); ro=$$?; \
	  if [ $$rr -ne 0 ] || [ $$ro -ne 0 ]; then echo "[FAIL] $$k (exit plain=$$rr $(VARIANT)=$$ro)"; fail=1; \
	  elif [ -z "$$ref" ]; then echo "[FAIL] $$k (no output)"; fail=1; \
	  elif [ "$$ref" = "$$out" ]; then echo "[SAME] $$k"; \
	  else echo "[DIFF] $$k"; fail=1; fi; \
	done; exit $$fail

list:
	@echo "kernels : $(KERNELS)"
	@echo "variants: $(VARIANTS)"
	@echo "out dir : $(OUT_DIR)"

outdir:
	@echo $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)

# Baseline-fidelity microbenchmarks (targets micro, micro-check, micro-pgo): benchmarks/micro/micro.mk
-include $(BENCH_DIR)/micro/micro.mk
