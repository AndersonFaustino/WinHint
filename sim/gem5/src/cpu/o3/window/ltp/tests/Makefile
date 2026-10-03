# Host unit tests of ltp_core.hh (B9 LTP decision logic; no gem5 needed).
#   make -C sim/gem5/src/cpu/o3/window/ltp/tests     build + run (-j1)
ROOT    := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../../../../../../..)
WINHINT_BUILD ?= $(ROOT)/build
OUT     := $(WINHINT_BUILD)/tests
HERE    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CXX     ?= g++
COVERAGE_FLAGS ?=   # set by tooling/coverage.sh (e.g. --coverage -O0)
BIN     := $(OUT)/test_ltp_core

.NOTPARALLEL:
all: run

$(BIN): $(HERE)/test_ltp_core.cc $(HERE)/../ltp_core.hh
	mkdir -p $(OUT)
	$(CXX) -std=c++17 -O1 -g -Wall -Wextra $(COVERAGE_FLAGS) -o $@ $(HERE)/test_ltp_core.cc

run: $(BIN)
	$(BIN)

clean:
	rm -f $(BIN)
.PHONY: all run clean
