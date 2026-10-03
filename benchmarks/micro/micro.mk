# =============================================================================
# benchmarks/micro/micro.mk — baseline-fidelity microbenchmarks (docs/reference/proposal.md §7)
#
# Included by benchmarks/Makefile (one `-include` line). The targets re-invoke
# that Makefile with BENCH_DIR pointing at benchmarks/micro/ and a separate
# OUT_ROOT, so the micro builds use exactly the same toolchain flags, pass
# flags and variants as the ML kernels, but never mix with them:
#
#   $(WINHINT_BUILD)/benchmarks/micro/<arch>/<variant>[<cfg>]/<micro kernel>
#
# Targets (ARCH=riscv by default; run inside the winhint env):
#   micro         build every MICRO_VARIANTS variant of every micro kernel
#   micro-check   build them and compare each variant's stdout with plain under
#                 qemu-riscv64, on both inputs (small and large) — bit identical
#   micro-pgo     VARIANT=pgo/oracle_hinted with MICRO_MAP_DIR (per-region maps
#                 written by sim/fidelity/fidelity.py), e.g.
#                   make -C benchmarks micro-pgo MICRO_HINTED=pgo MICRO_MAP_DIR=...
#   micro-list    print kernels, variants and output root
# Variables: MICRO_KERNELS, MICRO_VARIANTS (default plain oracle winhint jones
#   jones_full), MICRO_OUT_ROOT, MICRO_MAP_DIR, MICRO_HINTED (pgo|oracle_hinted).
# =============================================================================

MICRO_SRC      := $(BENCH_DIR)/micro
MICRO_OUT_ROOT ?= $(WINHINT_BUILD)/benchmarks/micro
MICRO_KERNELS  ?= $(sort $(basename $(notdir $(shell grep -l '^int main' $(MICRO_SRC)/*.c))))
MICRO_VARIANTS ?= plain oracle winhint jones jones_full
MICRO_INPUTS   ?= small large
MICRO_HINTED   ?= pgo
MICRO_MAP_DIR  ?=

MICRO_SUBMAKE = $(MAKE) --no-print-directory -f $(BENCH_DIR)/Makefile \
  BENCH_DIR=$(MICRO_SRC) WINHINT_ROOT=$(WINHINT_ROOT) WINHINT_BUILD=$(WINHINT_BUILD) \
  OUT_ROOT=$(MICRO_OUT_ROOT) ARCH=$(ARCH) OPT=$(OPT) UNROLL=$(UNROLL) TILED=$(TILED) \
  MACHINE=$(MACHINE) KERNELS="$(MICRO_KERNELS)"

.PHONY: micro micro-check micro-pgo micro-list

micro:
	@for v in $(MICRO_VARIANTS); do \
	  $(MICRO_SUBMAKE) VARIANT=$$v all || exit 1; \
	done

micro-check: micro
	@fail=0; for v in $(filter-out plain,$(MICRO_VARIANTS)); do \
	  for i in $(MICRO_INPUTS); do \
	    echo "== $$v ($$i)"; \
	    $(MICRO_SUBMAKE) VARIANT=$$v CHECK_INPUT=$$i check || fail=1; \
	  done; \
	done; exit $$fail

micro-pgo:
	@test -n "$(MICRO_MAP_DIR)" || { echo "[ERROR] micro-pgo needs MICRO_MAP_DIR=<dir with <kernel>.json maps>"; exit 1; }
	@$(MICRO_SUBMAKE) VARIANT=$(MICRO_HINTED) ORACLE_DIR=$(MICRO_MAP_DIR) PGO_DIR=$(MICRO_MAP_DIR) all

micro-list:
	@echo "micro kernels : $(MICRO_KERNELS)"
	@echo "micro variants: $(MICRO_VARIANTS)"
	@echo "micro out root: $(MICRO_OUT_ROOT)/$(ARCH)"
