//===- Plugin.cpp - WinHint pass-plugin registration ----------------------===//
//
//   clang -O2 -fplugin=WinHint.so -fpass-plugin=WinHint.so  (one line)
//         -mllvm -winhint-target=sim/machines/riscv_ooo.json ...
//   opt -load-pass-plugin=WinHint.so -passes='print<winhint-demand>' x.ll
//   opt -load-pass-plugin=WinHint.so -passes=winhint x.ll
//
// (-fplugin= makes clang dlopen the library before it parses -mllvm, so the
//  -winhint-* options are known; -fpass-plugin= registers the passes.)
//
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Pass-plugin entry point of WinHint.so.
 *
 * Registers the WindowDemandAnalysis, the pipeline names
 * `print<winhint-demand>`, `require<winhint-demand>` (function) and
 * `winhint` / `winhint-place` (module), and, unless -winhint-auto=false,
 * adds HintPlacementPass at the optimizer-last extension point (skipped in
 * ThinLTO/FullLTO pre-link).
 *
 * Usage:
 * @code
 *   clang -O2 -fplugin=WinHint.so -fpass-plugin=WinHint.so \
 *         -mllvm -winhint-target=sim/machines/riscv_ooo.json ...
 *   opt -load-pass-plugin=WinHint.so -passes='print<winhint-demand>' x.ll
 *   opt -load-pass-plugin=WinHint.so -passes=winhint x.ll
 * @endcode
 */
#include "HintPlacement.h"
#include "Options.h"
#include "WindowDemandAnalysis.h"

#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Plugins/PassPlugin.h") // LLVM >= 23
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif

using namespace llvm;
using namespace winhint;

/// @brief Register the analysis, the pipeline-parsing callbacks and the
///        optimizer-last hook with PB.
static void registerCallbacks(PassBuilder &PB) {
  PB.registerAnalysisRegistrationCallback(
      [](FunctionAnalysisManager &FAM) { FAM.registerPass([] { return WindowDemandAnalysis(); }); });

  PB.registerPipelineParsingCallback(
      [](StringRef Name, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "print<winhint-demand>") {
          FPM.addPass(WindowDemandPrinterPass(errs()));
          return true;
        }
        if (Name == "require<winhint-demand>") {
          FPM.addPass(RequireAnalysisPass<WindowDemandAnalysis, Function>());
          return true;
        }
        return false;
      });
  PB.registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "winhint" || Name == "winhint-place") {
          MPM.addPass(HintPlacementPass());
          return true;
        }
        return false;
      });

  // Run after the optimizer (loops are unrolled/vectorized, calls inlined),
  // so the model sees the code that will execute.
  PB.registerOptimizerLastEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel, ThinOrFullLTOPhase Phase) {
        if (!OptAutoRegister)
          return;
        if (Phase == ThinOrFullLTOPhase::ThinLTOPreLink ||
            Phase == ThinOrFullLTOPhase::FullLTOPreLink)
          return;
        MPM.addPass(HintPlacementPass());
      });
}

/// @brief Plugin descriptor queried by opt/clang when loading WinHint.so.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "WinHint", LLVM_VERSION_STRING, registerCallbacks};
}
