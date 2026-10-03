# hw/Makefile -- builds the WinHint real-hardware runtime and tools (host, conda env `winhint`).
#
#   micromamba run -n winhint make -C hw -j1
#
# Uses the env's C compiler ($CC from the conda activation hook; falls back to
# `cc`). Outputs only under $(WINHINT_BUILD) (default: <repo>/build):
#   $(LIBDIR)/libwinhint.a, libwinhint.so, winhint.h, bench_migration, test_affinity
#   $(TOOLDIR)/wh_measure, pie_daemon, phase_workload[_hinted|_nop], uarch_probe   (make static: static copies)
#
# make uarch-smoke    functional smoke of the µarch probe on one P and one E CPU (tiny ranges;
#                     NOT a measurement -- the full probe is run by run_hw_experiments.py --uarch)
SRC := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
WINHINT_ROOT  ?= $(abspath $(SRC)/..)
WINHINT_BUILD ?= $(WINHINT_ROOT)/build
LIBDIR  ?= $(WINHINT_BUILD)/libwinhint
TOOLDIR ?= $(WINHINT_BUILD)/hw
ifeq ($(origin CC),default)
CC := $(shell command -v x86_64-conda-linux-gnu-cc 2>/dev/null || echo cc)
endif
CFLAGS  ?= -O2
CFLAGS  += -g -Wall -Wextra -Wno-unused-parameter -std=gnu11 -fPIC -I$(SRC)/libwinhint

ifneq (,$(filter $(SRC)%,$(abspath $(LIBDIR)) $(abspath $(TOOLDIR))))
$(error build outputs must not go into the source tree hw/ (LIBDIR=$(LIBDIR) TOOLDIR=$(TOOLDIR)))
endif

LIBOBJ := $(LIBDIR)/obj/winhint.o $(LIBDIR)/obj/whutil.o

.PHONY: all lib tools static clean uarch-smoke
all: lib tools

lib: $(LIBDIR)/libwinhint.a $(LIBDIR)/libwinhint.so $(LIBDIR)/bench_migration $(LIBDIR)/test_affinity \
     $(LIBDIR)/winhint.h

tools: $(TOOLDIR)/wh_measure $(TOOLDIR)/pie_daemon $(TOOLDIR)/phase_workload $(TOOLDIR)/phase_workload_hinted \
       $(TOOLDIR)/phase_workload_nop $(TOOLDIR)/uarch_probe

$(LIBDIR)/obj/%.o: $(SRC)/libwinhint/%.c $(SRC)/libwinhint/whutil.h $(SRC)/libwinhint/winhint.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIBDIR)/libwinhint.a: $(LIBOBJ)
	ar rcs $@ $^

$(LIBDIR)/libwinhint.so: $(LIBOBJ)
	$(CC) -shared -o $@ $^

$(LIBDIR)/winhint.h: $(SRC)/libwinhint/winhint.h
	cp $< $@

$(LIBDIR)/bench_migration: $(SRC)/libwinhint/bench_migration.c $(LIBDIR)/obj/whutil.o
	$(CC) $(CFLAGS) -o $@ $^

$(LIBDIR)/test_affinity: $(SRC)/libwinhint/test_affinity.c $(LIBDIR)/libwinhint.a
	$(CC) $(CFLAGS) -o $@ $< -L$(LIBDIR) -l:libwinhint.a

$(TOOLDIR)/wh_measure: $(SRC)/tools/wh_measure.c $(LIBDIR)/obj/whutil.o
	@mkdir -p $(TOOLDIR)
	$(CC) $(CFLAGS) -o $@ $^

$(TOOLDIR)/pie_daemon: $(SRC)/baselines/r4_pie/pie_daemon.c $(LIBDIR)/obj/whutil.o
	@mkdir -p $(TOOLDIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TOOLDIR)/phase_workload: $(SRC)/tools/phase_workload.c
	@mkdir -p $(TOOLDIR)
	$(CC) $(CFLAGS) -o $@ $<

$(TOOLDIR)/phase_workload_hinted: $(SRC)/tools/phase_workload.c $(LIBDIR)/libwinhint.a
	@mkdir -p $(TOOLDIR)
	$(CC) $(CFLAGS) -DWITH_HINTS -o $@ $< -L$(LIBDIR) -l:libwinhint.a

$(TOOLDIR)/phase_workload_nop: $(SRC)/tools/phase_workload.c
	@mkdir -p $(TOOLDIR)
	$(CC) $(CFLAGS) -DWITH_NOP_HINTS -o $@ $<

$(TOOLDIR)/uarch_probe: $(SRC)/tools/uarch_probe.c $(LIBDIR)/obj/whutil.o
	@mkdir -p $(TOOLDIR)
	$(CC) $(CFLAGS) -o $@ $^

uarch-smoke: $(TOOLDIR)/uarch_probe
	$(TOOLDIR)/uarch_probe -c P -p rob,mlp -n 0:32:16 -k 2 -i 200 -m 16 -r 1 -q > /dev/null
	$(TOOLDIR)/uarch_probe -c E -p lb,sb -n 0:8:8 -i 200 -m 16 -r 1 -q > /dev/null
	@echo "uarch_probe smoke: OK (functional only)"

# Statically linked copies of the tools (portable smoke tests).
static: lib
	@mkdir -p $(TOOLDIR)/static
	$(CC) $(CFLAGS) -static -o $(TOOLDIR)/static/pie_daemon $(SRC)/baselines/r4_pie/pie_daemon.c $(SRC)/libwinhint/whutil.c -lm
	$(CC) $(CFLAGS) -static -o $(TOOLDIR)/static/wh_measure $(SRC)/tools/wh_measure.c $(SRC)/libwinhint/whutil.c
	$(CC) $(CFLAGS) -static -o $(TOOLDIR)/static/phase_workload $(SRC)/tools/phase_workload.c
	$(CC) $(CFLAGS) -static -DWITH_HINTS -o $(TOOLDIR)/static/phase_workload_hinted $(SRC)/tools/phase_workload.c -L$(LIBDIR) -l:libwinhint.a

clean:
	rm -rf $(LIBDIR) $(TOOLDIR)
