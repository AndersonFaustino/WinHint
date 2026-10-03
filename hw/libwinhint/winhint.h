/**
 * @file winhint.h
 * @brief WinHint runtime-call interface (docs/interfaces.md §2).
 *
 * With `-mllvm -winhint-emit=call` the WinHint pass emits calls to these
 * functions instead of ISA NOP hints. On real Intel hybrid silicon the
 * runtime (libwinhint) maps the advisory window size to P/E-core placement.
 *
 * Link (host, conda env `winhint`; WINHINT_ROOT/WINHINT_BUILD from the env hook):
 *   -I$WINHINT_ROOT/hw/libwinhint  -L$WINHINT_BUILD/libwinhint -l:libwinhint.a
 * (or -lwinhint for the shared library + LD_LIBRARY_PATH / -Wl,-rpath).
 *
 * Runtime configuration: environment variables, see docs/guide/hardware/index.md
 * (WINHINT_MODE=off|log|migrate|sondag, WINHINT_THRESHOLD, WINHINT_LOG, ...).
 *
 * All three functions are safe to call from several threads (they serialize on an
 * internal spin lock), but only the calling thread's affinity is ever changed.
 */
#ifndef WINHINT_H
#define WINHINT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Advisory: use at most a W-entry window from here on. W = 0 = release.
 *
 * In `migrate` mode, `w >= WINHINT_THRESHOLD` (default 192) requests the P-core
 * set, `0 < w < threshold` the E-core set, and `w == 0` the `WINHINT_RELEASE`
 * side (default: the original affinity). Requests pass through hysteresis
 * (`WINHINT_HYST`) and min-dwell rate limiting (`WINHINT_MIN_DWELL_US`) before
 * `sched_setaffinity` is called on the calling thread. In `log` mode the
 * decision is only recorded; in `off` mode only the call is counted; in
 * `sondag` mode the call is counted and its window ignored.
 *
 * @param[in] w Window size in entries; 0 releases the restriction.
 */
void __winhint_setwin(unsigned w);

/**
 * @brief Marks entry into static region `id` (0..63). Does not change the window.
 *
 * Closes the current accounting segment (per-region wall time, perf counters,
 * RAPL energy when `WINHINT_LOG` is set) and starts a new one attributed to
 * `id`. In `sondag` mode (R5) this is also where placement is decided: the
 * region's type is sampled on P and E, then pinned to the better core type.
 * Ids >= 64 are mapped to the internal "no region" bucket.
 *
 * @param[in] id Static region identifier, 0..63.
 */
void __winhint_region(unsigned id);

/**
 * @brief Flush logs now (also done automatically at exit). Optional.
 *
 * Closes the current accounting segment, (re)writes the `WINHINT_LOG` CSV and
 * its `<path>.summary.json`, and flushes the `WINHINT_TRACE` stream. No-op for
 * outputs that were not configured.
 */
void winhint_flush(void);

#ifdef __cplusplus
}
#endif
#endif
