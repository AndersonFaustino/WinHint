//===- HintPlacement.h - WinHint setwin placement ---------------*- C++ -*-===//
//
// Module pass. Modes (-winhint-mode):
//   setwin (model)  dynamic program over the region tree (loop nests and
//                   call-graph summaries) that places setwin(W) hints where
//                   the modeled benefit outweighs the switch cost.
//   regions         region(id) markers at the entry of every top-level loop
//                   nest (B1 oracle / B7 profiling builds).
//   from-json=<f>   region(id) marker + setwin(W) per region from a JSON map
//                   (oracle_hinted, B7 PGO).
//   off             nothing.
// Writes <kernel>.regions.json and <kernel>.winhint.json (stats) to
// -winhint-out-dir; schema in docs/reference/stats-schema.md.
//
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Module pass that places setwin / region hints in loop preheaders.
 *
 * Pipeline names `winhint` and `winhint-place`; also added automatically at
 * the optimizer-last extension point (-winhint-auto). See HintPlacement.cpp
 * for the placement dynamic program.
 */
#ifndef WINHINT_HINTPLACEMENT_H
#define WINHINT_HINTPLACEMENT_H

#include "llvm/IR/PassManager.h"

namespace winhint {

/**
 * @brief WinHint hint-placement module pass (-winhint-mode selects the mode).
 */
class HintPlacementPass : public llvm::PassInfoMixin<HintPlacementPass> {
public:
  /**
   * @brief Analyze every defined function, decide and emit the hints, and
   *        write the regions/stats JSON files.
   *
   * Regions (top-level loops) are numbered over functions sorted by name;
   * hints are inserted before the terminator of the loop preheader (created
   * if missing). Unknown -winhint-mode / -winhint-emit values and an
   * unreadable from-json map are fatal errors.
   *
   * @param M   Module to instrument.
   * @param MAM Module analysis manager (WindowDemandAnalysis, LoopAnalysis and
   *            DominatorTreeAnalysis are obtained through the function proxy).
   * @return PreservedAnalyses::all() if the IR was not modified (no hint
   *         emitted and no preheader inserted, e.g. -winhint-emit=none, or
   *         mode off), otherwise PreservedAnalyses::none().
   */
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
  /// @brief Required pass: never skipped by optnone or opt-bisect.
  static bool isRequired() { return true; }
};

} // namespace winhint

#endif
