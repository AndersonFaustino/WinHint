//===- Options.cpp - WinHint command-line options -------------------------===//
/**
 * @file
 * @brief Definitions (names, descriptions, defaults) of the `-winhint-*`
 *        options and the cached TargetModel loader.
 */
#include "Options.h"

using namespace llvm;

namespace winhint {

cl::opt<std::string> OptTarget("winhint-target",
                               cl::desc("Machine description JSON (sim/machines/*.json)"),
                               cl::init(""));
cl::opt<std::string> OptEmit("winhint-emit", cl::desc("Hint emission: asm (default), call, none"),
                             cl::init("asm"));
cl::opt<std::string>
    OptMode("winhint-mode",
            cl::desc("setwin (static W* model + DP placement, default; alias: model) | "
                     "regions (region(id) markers only) | "
                     "from-json=<file> (region marker + setwin per region from a map) | off"),
            cl::init("setwin"));
cl::opt<double> OptSwitchCost("winhint-switch-cost",
                              cl::desc("Window switch cost in cycles (0 = per switch model)"),
                              cl::init(0));
cl::opt<std::string> OptSwitchModel(
    "winhint-switch-model",
    cl::desc("Default switch cost model: gem5 (resize + drain) or pe (P/E-core migration)"),
    cl::init("gem5"));
cl::opt<double> OptMigrationUs("winhint-migration-us",
                               cl::desc("P/E migration cost in microseconds (switch model pe)"),
                               cl::init(50.0));
cl::opt<double> OptHysteresis("winhint-hysteresis",
                              cl::desc("Relative margin a switch must win by (hysteresis)"),
                              cl::init(0.25));
cl::opt<double> OptHintCost("winhint-hint-cost",
                            cl::desc("Cost of executing one hint, cycles (0 = 1 asm / 8 call)"),
                            cl::init(0));
cl::opt<double> OptEnergyWeight(
    "winhint-energy-weight",
    cl::desc("Relative power of the full window vs. the core (cost of oversizing)"),
    cl::init(0.15));
cl::opt<unsigned> OptMinRegionInsts(
    "winhint-min-region-insts",
    cl::desc("Never hint a loop whose estimated work per entry is below this (0 = 2*W_max)"),
    cl::init(0));
cl::opt<unsigned> OptUnknownTrip("winhint-unknown-trip",
                                 cl::desc("Trip count assumed for loops with unknown bounds"),
                                 cl::init(1000));
cl::opt<std::string> OptCPModel(
    "winhint-cp-model",
    cl::desc("Critical-path window term: ii (W_cp = CP * min(issue_width, body/II), "
             "default) or width (W_cp = CP * issue_width, the closed form of PROPOSAL §3.1)"),
    cl::init("ii"));
cl::opt<std::string> OptOutDir("winhint-out-dir",
                               cl::desc("Directory for <kernel>.regions.json / .winhint.json"),
                               cl::init(""));
cl::opt<std::string> OptStatsFile(
    "winhint-stats-file",
    cl::desc("Write the per-module stats JSON here (default <out-dir>/<kernel>.winhint.json)"),
    cl::init(""));
cl::opt<std::string> OptKernel("winhint-kernel",
                               cl::desc("Kernel name for output files (default: source stem)"),
                               cl::init(""));
cl::opt<bool> OptRegionMarkers("winhint-region-markers",
                               cl::desc("Also emit region(id) markers in model/from-json mode"),
                               cl::init(false));
cl::opt<bool> OptVerbose("winhint-verbose",
                         cl::desc("Print the target model and decisions to stderr"),
                         cl::init(false));
cl::opt<bool> OptAutoRegister(
    "winhint-auto",
    cl::desc("Run the WinHint pass at the optimizer-last extension point (default on)"),
    cl::init(true));

// Documented in Options.h.
const TargetModel &getTargetModel() { return getCachedTargetModel(OptTarget, "winhint"); }

} // namespace winhint
