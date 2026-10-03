# sim/tests — WinHint gem5 test programs (host, conda env `winhint`).
#   make -C sim/tests            -> $(WINHINT_BUILD)/sim_tests/
WINHINT_BUILD ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../build)
RVCC    ?= $(or $(WINHINT_RISCV_CC),riscv64-conda-linux-gnu-gcc)
OUT     ?= $(WINHINT_BUILD)/sim_tests
# RISC-V flags. Not CFLAGS: the conda env exports host (x86) CFLAGS such as -march=nocona,
# which the RISC-V cross compiler rejects.
RVCFLAGS ?= -static -O2 -Wall

all: $(OUT)/window_test $(OUT)/window_test_nohints

$(OUT)/window_test: window_test.c winhint_hint.h
	mkdir -p $(OUT)
	$(RVCC) $(RVCFLAGS) -o $@ window_test.c

$(OUT)/window_test_nohints: window_test.c winhint_hint.h
	mkdir -p $(OUT)
	$(RVCC) $(RVCFLAGS) -DWINHINT_NO_HINTS -o $@ window_test.c

clean:
	rm -rf $(OUT)
.PHONY: all clean

# Host unit tests of policy.hh + static/hint/hybrid (no gem5; stubs from the
# window/tests dir). Build + run.
WIN_DIR  := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../gem5/src/cpu/o3/window)
UNIT_BIN := $(OUT)/test_policies_b
# Absolute source path: gcov/gcovr (tooling/coverage_gate.py) resolve it from any directory.
UNIT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))/unit
HOSTCXX  ?= $(or $(CXX),g++)
COVERAGE_FLAGS ?=   # set by tooling/coverage.sh (e.g. --coverage -O0)
unit: $(UNIT_BIN)
	$(UNIT_BIN)
$(UNIT_BIN): unit/test_policies_b.cc $(WIN_DIR)/policy.hh $(WIN_DIR)/hint_policy.hh \
             $(WIN_DIR)/static_policy.cc $(WIN_DIR)/hint_policy.cc $(WIN_DIR)/hybrid_policy.cc
	mkdir -p $(OUT)
	$(HOSTCXX) -std=c++17 -O1 -g -Wall -Wextra -Wno-unused-parameter $(COVERAGE_FLAGS) \
	    -I$(WIN_DIR)/tests/stubs -I$(WIN_DIR)/../../.. -o $@ $(UNIT_DIR)/test_policies_b.cc \
	    $(WIN_DIR)/static_policy.cc $(WIN_DIR)/hint_policy.cc $(WIN_DIR)/hybrid_policy.cc
.PHONY: unit
