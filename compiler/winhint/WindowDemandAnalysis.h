//===- WindowDemandAnalysis.h - WinHint window-demand model -----*- C++ -*-===//
//
// Per-loop (region) out-of-order window demand:
//
//   L_mem    expected service latency of the long-latency load classes
//            (footprint / reuse distance vs. cache capacities, memory latency)
//   D_indep  dynamic instructions between independent long-latency loads
//            (iteration size incl. unrolled copies and inner loops / misses per
//            iteration, loads on
//            a loop-carried recurrence (pointer chase, DA flow dep) excluded)
//   CP       critical path of one iteration's dependence DAG (cycles)
//
//   MLP_target = min(L1D MSHRs, L_mem * rate / D_indep)   (fractional)
//   rate       = min(issue_width, body / II)  (II: recurrences, width, dividers)
//   W*  = min(W_max, max(ceil(MLP_target * D_indep), W_cp))
//   W_cp = ceil(CP * rate)     (= CP * issue_width when the loop is issue-bound;
//                               -winhint-cp-model=width uses CP * issue_width)
//
// W* is mapped to the smallest window configuration with ROB >= W*
// (docs/interfaces.md §3).
//
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Function analysis computing the per-loop out-of-order window demand
 *        W* (cost model of PROPOSAL §3.1), plus its printer pass and the
 *        execution-cost function shared with HintPlacement.
 *
 * The formulas are summarized in the comment block above.
 * TargetModel::MLPTargetOverride, if set, replaces the derived MLP_target.
 * Pointer-chase and loop-carried loads are excluded from the miss count; a
 * loop whose only long-latency loads are chases gets MLP_target = 1 and
 * W_mlp = body size.
 */
#ifndef WINHINT_WINDOWDEMANDANALYSIS_H
#define WINHINT_WINDOWDEMANDANALYSIS_H

#include "TargetModel.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include <memory>
#include <string>
#include <vector>

namespace winhint {

/// Address pattern of a memory access with respect to its own loop.
enum class AccessClass {
  Streaming, ///< affine, inner stride smaller than a cache line
  Strided,   ///< affine, inner stride >= one cache line
  Invariant, ///< address invariant in the own loop
  Indirect,  ///< not affine in the own loop and not a chase (e.g. A[B[i]])
  Chase      ///< address computed from a load on a header-PHI recurrence (pointer chase)
};
/// @brief Short name used in printer output: "stream", "strided", "invariant",
///        "indirect" or "chase".
const char *accessClassName(AccessClass C);

/// A group of accesses with the same base and strides whose start addresses
/// differ by small constants (unrolled copies, A[i] / A[i+1], ...).
/// Only affine (Streaming/Strided/Invariant) accesses are merged; indirect
/// and chase accesses always form singleton groups.
struct AccessGroup {
  const llvm::Instruction *Leader = nullptr; ///< first load/store of the group
  unsigned Members = 1;                      ///< number of grouped accesses
  bool IsLoad = true;                        ///< loads and stores are grouped separately
  AccessClass Class = AccessClass::Streaming; ///< pattern of the leader
  double InnerStride = 0; ///< bytes per innermost iteration (|s0|)
  bool StrideKnown = true; ///< false if the own-loop stride is symbolic (assumed 64 lines)
  double ElemBytes = 4;    ///< store size of the leader's type
  double SpanBytes = 4; ///< address span of the members in one iteration
  double FootprintBytes = 0;      ///< over one entry of the outermost loop
  double ReuseDistanceBytes = -1; ///< < 0: no reuse inside the nest
  unsigned Level = 0;             ///< level serving the misses (0 = L1)
  double MissesPerIter = 0;       ///< misses beyond L1 per innermost iteration
  bool LoopCarried = false;       ///< on a memory recurrence (DA flow dep)
  double ObjectBytes = -1;        ///< size of the underlying object if known
  // Internal: per-ancestor strides and footprints (index 0 = own loop).
  std::vector<double> Strides;     ///< |stride| in bytes per iteration of ancestor k
  std::vector<bool> StrideIsZero;  ///< address invariant in ancestor k
  std::vector<double> FootprintAt; ///< footprint over one *entry* of ancestor k
};

/// One loop of the region tree. Top-level loops are the regions of
/// docs/interfaces.md §5.
struct LoopDemand {
  llvm::Loop *L = nullptr;            ///< the IR loop
  LoopDemand *Parent = nullptr;       ///< enclosing loop, nullptr for a top-level loop
  std::vector<LoopDemand *> Children; ///< program order
  unsigned Depth = 1;                 ///< loop depth (1 = top level)
  unsigned Line = 0;                  ///< source line of the loop start (0 if no debug info)
  std::string HeaderName;             ///< header block name, else "loop<index>"
  unsigned IndexInFunction = 0; ///< preorder index within the function

  bool TripKnown = false; ///< exact constant backedge-taken count
  bool TripBounded = false; ///< upper bound only (call-site argument values or SCEV max)
  double Trip = 1;           ///< iterations per entry
  double OwnInsts = 0;       ///< per iteration, own blocks (excl. subloops)
  double DynInsts = 0;       ///< per entry, incl. subloops (excl. callees)
  double NestFootprint = 0;  ///< bytes over one entry, all accesses in the nest

  std::vector<AccessGroup> Groups; ///< accesses of the own blocks
  /// Calls to functions defined in the module made from own blocks, with
  /// their RPO position (for ordering).
  std::vector<std::pair<const llvm::CallBase *, unsigned>> Calls;
  unsigned RPOIndex = 0; ///< RPO index of the header block (program order)

  // Model outputs.
  /// L_mem: mean service latency of the independent long-latency loads
  /// (cycles); MissesPerIter: their misses per iteration; DIndep: D_indep in
  /// dynamic instructions (INFINITY without such misses); MLPTarget: fractional
  /// MLP target.
  double Lmem = 0, MissesPerIter = 0, DIndep = 0, MLPTarget = 0;
  /// CP: critical path of one iteration of the own blocks (cycles); RecMII:
  /// longest register recurrence through a header PHI (cycles); II: initiation
  /// interval max(RecMII, body/issue_width, divider occupancy, 1).
  double CP = 0, RecMII = 0, II = 1;
  /// W_mlp, W_cp, W* (entries) and the selected window config index.
  unsigned Wmlp = 0, Wcp = 0, WStar = 0, Config = 0;
  /// Counts over the own blocks: loads, stores, independent long-latency load
  /// groups, chase groups, indirect groups, loads on a memory recurrence.
  unsigned NumLoads = 0, NumStores = 0, NumLongLat = 0, NumChase = 0, NumIndirect = 0,
           NumLCD = 0;
  unsigned UnrollFactor = 1;  ///< largest group size (proxy for the unroll factor)
  bool Conservative = false;  ///< unknown trip count or symbolic inner stride
  bool MemoryBound = false;   ///< L_mem * rate exceeds the body size (or a chase)
  std::string Notes;          ///< free-form notes (e.g. "unknown-trip ")

  // Nest aggregates (meaningful for top-level loops).
  double NestWStar = 0;   ///< dynamic-instruction weighted W* over the nest
  unsigned NestConfig = 0; ///< best uniform config for the nest (cost model)
};

/// Result of WindowDemandAnalysis for one function.
struct FunctionDemand {
  llvm::Function *F = nullptr; ///< analyzed function
  std::vector<std::unique_ptr<LoopDemand>> Loops; ///< preorder
  std::vector<LoopDemand *> TopLevel;             ///< program order
  llvm::DenseMap<const llvm::Loop *, LoopDemand *> ByLoop; ///< IR loop -> demand
  double StraightInsts = 0; ///< instructions outside any loop
  /// Calls to defined functions outside any loop, with their block's RPO index.
  std::vector<std::pair<const llvm::CallBase *, unsigned>> StraightCalls;

  /// @brief New-PM invalidation hook: the result is invalidated unless
  ///        WindowDemandAnalysis (or all function analyses) is preserved.
  bool invalidate(llvm::Function &F, const llvm::PreservedAnalyses &PA,
                  llvm::FunctionAnalysisManager::Invalidator &Inv);
  /// @brief Print the per-loop model (trip, body, groups, and one
  ///        `WINHINT fn=... W*=...` line per loop) as used by the lit tests.
  void print(llvm::raw_ostream &OS, const TargetModel &TM) const;
};

/**
 * @brief Execution cost (cycle-equivalents, energy-weighted) of running Insts
 *        dynamic instructions whose demand is WStar under window config C.
 *
 * Cost = Insts / (IPC0 * min(1, ROB_C / WStar)) * (1 + EnergyWeight * ROB_C / W_max),
 * with IPC0 = max(1, issue_width / 2). An undersized window slows execution
 * proportionally; an oversized one pays the energy term.
 *
 * @param Insts        Dynamic instructions (<= 0 gives cost 0).
 * @param WStar        Window demand in entries (<= 0 is treated as fully served).
 * @param C            Config index (clamped to the table).
 * @param TM           Target model (non-empty window table, as guaranteed by
 *                     getCachedTargetModel()).
 * @param EnergyWeight Relative power of the full window (-winhint-energy-weight).
 * @return Cost in cycle-equivalents.
 */
double execCost(double Insts, double WStar, unsigned C, const TargetModel &TM,
                double EnergyWeight);

/// Summary of a library call (libm) used inside the dependence DAG.
struct LibCallInfo {
  double Insts;   ///< dynamic instructions per call
  /// Latency in cycles; -1 = use OpLatencies::FpSqrt (and occupy an FP
  /// divider), -2 = use OpLatencies::FpFma.
  double Latency;
};
/**
 * @brief Look up a summarized library routine (expf, logf, tanhf, sqrt, ...).
 *
 * `llvm.<name>.<type>` intrinsics are mapped to the libm name, with an "f"
 * suffix when the (element) type is f32 (e.g. llvm.exp.f32 and
 * llvm.exp.v4f32 -> expf, llvm.exp.v2f64 -> exp).
 *
 * @param[in]  Name Callee name.
 * @param[out] Info Instruction count and latency when found.
 * @return true if Name is in the table.
 */
bool getLibCallInfo(llvm::StringRef Name, LibCallInfo &Info);

/**
 * @brief New-PM function analysis producing a FunctionDemand.
 *
 * Requires LoopAnalysis, ScalarEvolutionAnalysis and DependenceAnalysis and
 * uses the target model of -winhint-target. Declarations yield an empty result.
 */
class WindowDemandAnalysis : public llvm::AnalysisInfoMixin<WindowDemandAnalysis> {
  friend llvm::AnalysisInfoMixin<WindowDemandAnalysis>;
  static llvm::AnalysisKey Key; ///< analysis identity for the pass manager

public:
  using Result = FunctionDemand; ///< analysis result type
  /// @brief Build the loop tree of F and compute the window demand of every loop.
  ///
  /// Reports a fatal error for an unknown -winhint-cp-model value.
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

/// Printer pass registered as `print<winhint-demand>`.
class WindowDemandPrinterPass : public llvm::PassInfoMixin<WindowDemandPrinterPass> {
  llvm::raw_ostream &OS; ///< output stream (stderr when registered by the plugin)

public:
  /// @brief Print to OS.
  explicit WindowDemandPrinterPass(llvm::raw_ostream &OS) : OS(OS) {}
  /// @brief Print the FunctionDemand of F (skips declarations); preserves all.
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
  /// @brief Required pass: never skipped by optnone or opt-bisect.
  static bool isRequired() { return true; }
};

} // namespace winhint

#endif
