//===- Options.h - WinHint command-line options -----------------*- C++ -*-===//
/**
 * @file
 * @brief Global `-winhint-*` LLVM command-line options of the WinHint plugin
 *        and access to the selected TargetModel.
 *
 * With clang the options are passed as `-mllvm -winhint-...` (the plugin must
 * also be loaded with -fplugin= so that they are registered before parsing).
 * Defaults and descriptions are in Options.cpp.
 */
#ifndef WINHINT_OPTIONS_H
#define WINHINT_OPTIONS_H

#include "TargetModel.h"
#include "llvm/Support/CommandLine.h"
#include <string>

namespace winhint {

/// -winhint-target=<json>: machine description (sim/machines/*.json); empty = builtin default.
extern llvm::cl::opt<std::string> OptTarget;        // -winhint-target=<json>
/// -winhint-emit=asm|call|none: hint materialization (default asm), see EmitMode.
extern llvm::cl::opt<std::string> OptEmit;          // -winhint-emit=asm|call|none
/// -winhint-mode=setwin|regions|from-json=<f>|off (default setwin; alias model).
extern llvm::cl::opt<std::string> OptMode;          // -winhint-mode=setwin|regions|from-json=<f>|off
/// -winhint-switch-cost: window switch cost in cycles (0 = derive from the switch model).
extern llvm::cl::opt<double> OptSwitchCost;         // -winhint-switch-cost=<cycles>
/// -winhint-switch-model=gem5|pe: default switch cost model (resize+drain or P/E migration).
extern llvm::cl::opt<std::string> OptSwitchModel;   // -winhint-switch-model=gem5|pe
/// -winhint-migration-us: P/E migration cost in microseconds (default 50, switch model pe).
extern llvm::cl::opt<double> OptMigrationUs;        // -winhint-migration-us
/// -winhint-hysteresis: relative margin a switch must win by (default 0.25).
extern llvm::cl::opt<double> OptHysteresis;         // -winhint-hysteresis
/// -winhint-hint-cost: cycles per executed hint (0 = 1 in asm mode, 8 in call mode).
extern llvm::cl::opt<double> OptHintCost;           // -winhint-hint-cost
/// -winhint-energy-weight: relative power of the full window vs. the core (default 0.15).
extern llvm::cl::opt<double> OptEnergyWeight;       // -winhint-energy-weight
/// -winhint-min-region-insts: minimum work per entry for a hinted loop (0 = 2 * W_max).
extern llvm::cl::opt<unsigned> OptMinRegionInsts;   // -winhint-min-region-insts
/// -winhint-unknown-trip: trip count assumed for loops with unknown bounds (default 1000).
extern llvm::cl::opt<unsigned> OptUnknownTrip;      // -winhint-unknown-trip
/// -winhint-cp-model=ii|width: critical-path window term (default ii).
extern llvm::cl::opt<std::string> OptCPModel;       // -winhint-cp-model=ii|width
/// -winhint-out-dir: directory for <kernel>.regions.json and <kernel>.winhint.json.
extern llvm::cl::opt<std::string> OptOutDir;        // -winhint-out-dir
/// -winhint-stats-file: explicit stats JSON path (default <out-dir>/<kernel>.winhint.json).
extern llvm::cl::opt<std::string> OptStatsFile;     // -winhint-stats-file
/// -winhint-kernel: kernel name for output files (default: source file stem).
extern llvm::cl::opt<std::string> OptKernel;        // -winhint-kernel
/// -winhint-region-markers: also emit region(id) markers in setwin/from-json mode.
extern llvm::cl::opt<bool> OptRegionMarkers;        // -winhint-region-markers
/// -winhint-verbose: print the target model and the placement decisions to stderr.
extern llvm::cl::opt<bool> OptVerbose;              // -winhint-verbose
/// -winhint-auto: add the placement pass at the optimizer-last extension point (default on).
extern llvm::cl::opt<bool> OptAutoRegister;         // -winhint-auto (pipeline hook)

/**
 * @brief Target model selected by -winhint-target.
 *
 * Loaded lazily and cached per path through getCachedTargetModel()
 * (thread-safe; a new option value loads that file). On a load error a
 * warning is printed and the builtin default machine is used; an empty
 * window table is a fatal error, so TargetModel::Window is never empty.
 *
 * @return Reference to the cached model (valid for the process lifetime).
 */
const TargetModel &getTargetModel();

} // namespace winhint

#endif
