# Host unit tests for the B2-B5 window policies (no gem5 needed).
#   make -C sim/gem5/src/cpu/o3/window/tests          build + run
# Output: $(WINHINT_BUILD)/tests/test_window_policies. Single job (-j1).
ROOT    := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../../../../../..)
WINHINT_BUILD ?= $(ROOT)/build
OUT     := $(WINHINT_BUILD)/tests
CXX     ?= g++
CXXFLAGS ?= -O1 -g -Wall -Wextra -Wno-unused-parameter
COVERAGE_FLAGS ?=   # set by tooling/coverage.sh (e.g. --coverage -O0)
SRC     := $(ROOT)/sim/gem5/src
WIN     := $(SRC)/cpu/o3/window
POLICIES := $(WIN)/occupancy_policy.cc $(WIN)/mlp_policy.cc \
            $(WIN)/bbv_policy.cc $(WIN)/lut_policy.cc
BIN     := $(OUT)/test_window_policies

.NOTPARALLEL:
all: run

$(BIN): $(WIN)/tests/test_policies.cc $(POLICIES) $(wildcard $(WIN)/*.hh) \
        $(WIN)/tests/stubs/base/logging.hh $(WIN)/tests/stubs/base/types.hh
	mkdir -p $(OUT)
	$(CXX) -std=c++17 $(CXXFLAGS) $(COVERAGE_FLAGS) -Wall -Wextra -Wno-unused-parameter -I$(WIN)/tests/stubs -I$(SRC) -o $@ \
	    $(WIN)/tests/test_policies.cc $(POLICIES)

run: $(BIN)
	mkdir -p $(OUT)/tmp
	$(BIN) $(ROOT)/sim/tests/test.lut $(OUT)/tmp

clean:
	rm -f $(BIN)
.PHONY: all run clean
