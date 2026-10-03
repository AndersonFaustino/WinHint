//===- TargetModel.h - WinHint machine description --------------*- C++ -*-===//
//
// Machine parameters consumed by the WinHint cost model and the baselines.
// Loaded from sim/machines/*.json (docs/interfaces.md §3 and §5). Every field
// is optional; missing fields fall back to the defaults below, which describe
// the default gem5 RISC-V O3 machine (riscv_ooo.json).
//
// API stability: this header (and HintEmitter.h) is shared with the baselines
// in compiler/baselines/ (B6 JonesIQ uses TargetModel::{loadJSON, Lat,
// IssueWidth, DispatchWidth, CommitWidth, Window, l1Latency, configForIQ,
// configCovering, configForW, Name} and getCachedTargetModel()). Fields and
// signatures are only ever added, never renamed or removed; behavior changes
// are noted here:
//   - loadJSON() now applies a "window" section that yields no configs (the
//     table becomes empty instead of silently keeping the defaults), and
//     getCachedTargetModel() rejects an empty table with a fatal error, so
//     every model it returns has a non-empty Window.
// Link against the CMake target WinHintCommon.
//
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Machine description (core widths, caches, memory, op latencies,
 *        window configuration table) for the WinHint cost model and baselines.
 *
 * A TargetModel is default-constructed with the parameters of the default gem5
 * RISC-V O3 machine (riscv_ooo.json) and optionally overridden field by field
 * from a machine JSON file with TargetModel::loadJSON(). The window table
 * follows docs/interfaces.md §3; config indices returned by the lookup helpers
 * are indices into TargetModel::Window (ascending ROB).
 */
#ifndef WINHINT_TARGETMODEL_H
#define WINHINT_TARGETMODEL_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>
#include <vector>

namespace winhint {

/// One window configuration (docs/interfaces.md §3): ROB, IQ, LQ and SQ
/// entry counts that are scaled together.
struct WindowConfig {
  unsigned ROB, IQ, LQ, SQ; ///< entries in the ROB, issue queue, load queue, store queue
};

/// One data-cache level (L1D, L2, optional L3).
struct CacheLevel {
  std::string Name;   ///< "L1D", "L2" or "L3"
  uint64_t SizeBytes; ///< capacity in bytes
  unsigned Latency; ///< load-to-use latency in cycles when served by this level
};

/// Per-operation latencies (cycles) for the dependence-DAG critical path.
/// Each field can be overridden from the machine JSON ("latency" section or
/// cpu.latency_cycles, keys int_alu, int_mul, ..., call).
struct OpLatencies {
  unsigned IntAlu = 1, IntMul = 3, IntDiv = 20; ///< integer ALU / multiply / divide
  unsigned FpAdd = 4, FpMul = 4, FpFma = 5, FpDiv = 12, FpSqrt = 24, FpCvt = 3; ///< FP ops
  unsigned Store = 1, Branch = 1, Call = 5; ///< store, branch, call overhead
};

/**
 * @brief Machine parameters consumed by the window-demand model, the hint
 *        placement and the baselines.
 *
 * Defaults describe the default gem5 RISC-V O3 machine; loadJSON() overrides
 * only the fields present in the file.
 */
struct TargetModel {
  std::string Name = "builtin-default"; ///< JSON "name", else the file path
  std::string SourcePath;               ///< path given to loadJSON(), empty for defaults

  // Core.
  unsigned IssueWidth = 4;    ///< instructions issued per cycle
  unsigned DispatchWidth = 4; ///< dispatch (or decode) width; defaults to IssueWidth in loadJSON()
  unsigned CommitWidth = 4;   ///< instructions committed per cycle
  double ClockGHz = 2.0;      ///< core clock, used to convert memory latency_ns to cycles

  // Memory hierarchy.
  unsigned LineSize = 64; ///< cache line size in bytes
  std::vector<CacheLevel> Caches; ///< ordered L1D, L2, [L3]
  unsigned MemLatency = 150;      ///< cycles, L2 miss to DRAM and back
  unsigned L1DMSHRs = 16;         ///< bounds the useful MLP
  bool StridePrefetcher = false;  ///< unit-stride streams hidden down to L2
  double CacheEffectiveFraction = 0.75; ///< usable share of each capacity

  OpLatencies Lat;
  /// Non-pipelined units (gem5 FUPool: FP_MultDiv and IntMultDiv, count 2):
  /// divides and square roots occupy one of them for their full latency.
  unsigned FpDivUnits = 2, IntDivUnits = 2;

  /// Window configuration table, sorted by ascending ROB.
  std::vector<WindowConfig> Window;

  // WinHint model knobs that may be set per machine ("winhint" section).
  double MLPTargetOverride = 0;   ///< 0 = derive from latency and MSHRs
  double SwitchCostCycles = 0;    ///< 0 = derive (drain of the largest ROB)

  /// @brief Initialize the default caches (L1D 32 KiB/4 cyc, L2 256 KiB/14 cyc)
  ///        and the four-entry window table of docs/interfaces.md §3.
  TargetModel();

  /**
   * @brief Parse a machine JSON file and override the fields it specifies.
   *
   * Recognized sections: "name", "cpu" (issue/dispatch/decode/commit width,
   * clock, latency_cycles, fu), "cache" (flat l1d_size/l1d_latency/... keys,
   * gem5 *_tag/_data/_response_latency parts, or nested "l1d"/"l2"/"l3"
   * objects; prefetcher; mshrs; effective_fraction), "memory" (latency_cycles,
   * latency_ns or latency), "latency", "fu", "window" (object with "configs"
   * or parallel "rob"/"iq"/"lq"/"sq" arrays, or an array of objects; IQ/LQ/SQ
   * default to ROB/2, ROB/4, ROB/4) and "winhint" (mlp_target,
   * switch_cost_cycles, memory_latency_cycles, l1d_mshrs).
   * A "window" section without any valid config (no "rob") leaves Window
   * empty; getCachedTargetModel() rejects such a model. Without a "window"
   * section the default table is kept.
   *
   * @param[in]  Path Path of the JSON file.
   * @param[out] Err  Error message when the file cannot be read or parsed, or
   *                  the top-level value is not an object.
   * @return true on success, false on failure (the model is left unchanged).
   */
  bool loadJSON(llvm::StringRef Path, std::string &Err);

  /// @brief Largest ROB size in the window table (256 if the table is empty).
  unsigned wMax() const;
  /// @brief L1D load-to-use latency in cycles (4 if no cache level is defined).
  unsigned l1Latency() const { return Caches.empty() ? 4 : Caches[0].Latency; }

  /// @brief Smallest config with ROB >= W; the largest if none fits or W == 0.
  /// @param W Window demand in ROB entries.
  /// @return Index into Window (0 if the table is empty).
  unsigned configForW(unsigned W) const;
  /// @brief Smallest config whose ROB/IQ/LQ/SQ all cover the given demands.
  /// @return Index into Window; the largest config if none covers them, 0 if empty.
  unsigned configCovering(unsigned ROB, unsigned IQ, unsigned LQ, unsigned SQ) const;
  /// @brief Smallest config with IQ >= the demand.
  /// @return Index into Window; the largest config if none fits, 0 if empty.
  unsigned configForIQ(unsigned IQ) const;
  /// @brief Service latency (cycles) for a level index.
  /// @param Level Index into Caches; Caches.size() (or larger) means memory,
  ///              whose latency is the last cache latency plus MemLatency.
  unsigned levelLatency(unsigned Level) const;
  /// @brief Name of a level index ("L1D", "L2", ...; "MEM" past the caches).
  std::string levelName(unsigned Level) const;
  /// @brief Default switch cost for the gem5 mechanism, in cycles.
  ///
  /// Returns SwitchCostCycles if set, else 10 + wMax() / 2 / CommitWidth
  /// (drain of half the largest ROB plus a refill allowance).
  double gem5SwitchCost() const;

  /// @brief Print a human-readable summary (widths, caches, latencies, units,
  ///        knobs, window table); -winhint-verbose prints it.
  void print(llvm::raw_ostream &OS) const;
};

/**
 * @brief Machine model for a machine JSON path, loaded once and cached.
 *
 * Thread-safe; one model per distinct path, kept for the lifetime of the
 * process (the returned reference stays valid). An empty path gives the
 * builtin default machine. If the file cannot be read or parsed, a warning
 * prefixed with Tool is printed and the builtin default machine is used. A
 * model whose window table is empty is a fatal configuration error, so
 * callers may rely on a non-empty TargetModel::Window.
 *
 * @param Path Machine JSON (a sim/machines JSON file), or empty.
 * @param Tool Prefix of the diagnostics ("winhint", "jones-iq").
 * @return Reference to the cached model.
 */
const TargetModel &getCachedTargetModel(const std::string &Path, llvm::StringRef Tool);

/// @brief Parse "32kB", "2MB", "512MB", "1GiB", "4096" into bytes.
///
/// Suffixes are case-insensitive and binary (k/kb/kib = 1024, m/mb/mib,
/// g/gb/gib); no suffix or "b" means bytes.
/// @param S       Size string.
/// @param Default Value returned for an empty string, a missing number or an
///                unknown suffix.
/// @return Size in bytes.
uint64_t parseSize(llvm::StringRef S, uint64_t Default);
/// @brief Parse "2GHz", "2.5GHz", "800MHz" into GHz.
/// @param S       Clock string; a bare number is taken as GHz.
/// @param Default Value returned if no number is found or the unit is unknown.
/// @return Clock frequency in GHz.
double parseClockGHz(llvm::StringRef S, double Default);

} // namespace winhint

#endif
