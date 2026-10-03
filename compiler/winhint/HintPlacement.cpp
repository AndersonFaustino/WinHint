//===- HintPlacement.cpp - WinHint setwin placement -----------------------===//
//
// Placement model
// ---------------
// States are the window configurations 0..K-1 plus U ("unknown", e.g. at a
// function entry; work in U is charged the worst case over all configurations
// and leaving U always costs a switch). Each node of the region tree is a loop; the body of
// a loop (and the top level of a function) is a sequence of items in program
// order: the node's own instructions, child loops and calls.
//
// G(N, s) = cheapest way to execute one entry of loop N entered in state s,
// together with the state on exit. Inside N the body runs Trip(N) times, so
// we enumerate the steady state e at the top of each iteration and run a
// sequence DP over the items that must return to e at the end of the body
// (cyclic consistency). For a child loop C entered in state t the options are
//   no hint      : G(C, t)
//   setwin(c)    : h + S_eff*[c != t] + G(C, c)       (hint in C's preheader)
// where h is the cost of executing a hint and S_eff = S * (1 + hysteresis) is
// the switch cost inflated by the hysteresis margin. Calls use call-graph
// summaries (callees are solved first): a callee without hints costs its
// uniform cost in the current state; a callee with hints costs its own
// optimum and leaves the window in its exit state. libm calls are summarized
// inside the dependence DAG by WindowDemandAnalysis.
//
// Hoisting falls out of the DP: a hint placed on an outer loop executes once
// per entry of that loop, a hint on an inner loop once per inner entry, and
// a child whose best state equals the inherited one gets no hint (redundant
// hints are never emitted). Loops whose work per entry is below
// -winhint-min-region-insts are never hinted.
//
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief HintPlacementPass: region numbering, the setwin placement dynamic
 *        program (Placer), region-map loading for from-json mode, hint
 *        emission and the regions/stats JSON outputs.
 *
 * The placement model is described in the comment block above. Outputs:
 * `<kernel>.regions.json` in -winhint-out-dir and the stats JSON
 * (`winhint-stats/1`, docs/reference/stats-schema.md) in -winhint-out-dir
 * or at -winhint-stats-file; nothing is written if neither is set.
 */
#include "HintPlacement.h"
#include "HintEmitter.h"
#include "Options.h"
#include "WindowDemandAnalysis.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <array>
#include <chrono>
#include <functional>
#include <cmath>
#include <map>
#include <set>

using namespace llvm;

namespace winhint {

namespace {

/// Placement mode parsed from -winhint-mode.
enum class Mode {
  Model,    ///< setwin / model: DP placement of setwin hints
  Regions,  ///< region(id) markers only
  FromJSON, ///< region marker + setwin per region from a JSON map
  Off       ///< do nothing
};

/// Call-graph summary of a solved function, used at its call sites.
struct FuncSummary {
  bool Done = false;           ///< set once solveFunction() has run
  double Insts = 50;           ///< dynamic instructions per call, incl. callees
  std::vector<double> Uniform; ///< cost per config with no internal hints
  double DemandW = 0;          ///< weighted W* (report only)
  bool HasHints = false;       ///< placement emits hints in it or leaves a known state
  double HintedCost = 0;       ///< optimal cost from the unknown entry state
  int Exit = -1; ///< config after return when HasHints (-1 = unknown)
};

/// One setwin decision: hint config Config in the preheader of loop D.
struct Decision {
  LoopDemand *D;   ///< loop that receives the hint
  unsigned Config; ///< index into TargetModel::Window
};

/// One element of a loop body (or function top level) in program order.
struct Item {
  /// Own instructions, a child loop, or a call to a defined function.
  enum Kind { Own, LoopK, CallK } K;
  LoopDemand *L = nullptr;      ///< LoopK: the child loop
  const CallBase *CB = nullptr; ///< CallK: the call
  double Insts = 0; ///< Own: instructions (per iteration / per call)
  double W = 1;     ///< Own: demand
  unsigned Pos = 0; ///< RPO position, used to order loops and calls
};

/**
 * @brief Dynamic program over the region tree that chooses setwin hints.
 *
 * States 0..K-1 are window configs, state U == K is "unknown". Costs are in
 * execCost() units; results are memoized per (loop, entry state).
 */
class Placer {
public:
  /// @param TM       Target model (window table).
  /// @param Sums     Function summaries, shared across solveFunction() calls.
  /// @param S        Switch cost in cycles.
  /// @param Hyst     Hysteresis margin; the effective switch cost is S * (1 + Hyst).
  /// @param H        Cost of executing one hint, in cycles.
  /// @param EW       Energy weight passed to execCost().
  /// @param MinInsts Loops with less work per entry (incl. calls) are never hinted.
  Placer(const TargetModel &TM, DenseMap<const Function *, FuncSummary> &Sums, double S,
         double Hyst, double H, double EW, double MinInsts)
      : TM(TM), K(TM.Window.size()), U(K), Sums(Sums), Seff(S * (1 + Hyst)), H(H),
        EW(EW), MinInsts(MinInsts) {}

  /// @brief Solve one function from the unknown state; fill its summary in
  ///        Sums and append its setwin decisions to Out. Callees must be solved first.
  void solveFunction(FunctionDemand &FD, std::vector<Decision> &Out);

  /// @brief Dynamic instructions per entry of N, including subloops and the
  ///        summarized instructions of calls (memoized).
  double instsWithCalls(LoopDemand *N);
  /// @brief Cost of one entry of N with the whole nest run in config C and no
  ///        hints (memoized).
  double uniformLoop(LoopDemand *N, unsigned C);

private:
  const TargetModel &TM;                         ///< target model
  const unsigned K, U;                           ///< number of configs; unknown state (== K)
  DenseMap<const Function *, FuncSummary> &Sums; ///< callee summaries
  const double Seff, H, EW, MinInsts;            ///< see constructor (Seff = S * (1 + Hyst))

  /// Result of G(N, s) or of a sequence DP.
  struct GRes {
    double Cost = INFINITY; ///< total cost; INFINITY = infeasible
    int Exit = 0;           ///< state on exit
    std::vector<int> Choice;     ///< per item: -1 none, else hint config
    std::vector<int> EntryState; ///< per item: state the item starts in
  };
  std::map<std::pair<LoopDemand *, unsigned>, GRes> Memo;     ///< solveLoop() results
  DenseMap<LoopDemand *, double> InstsMemo;                   ///< instsWithCalls() results
  std::map<std::pair<LoopDemand *, unsigned>, double> UniMemo; ///< uniformLoop() results
  std::map<LoopDemand *, std::vector<Item>> ItemsMemo; // stable references

  /// Cost of work in state St. The unknown state U (function entry: the
  /// caller's window is not known statically) is charged the worst case over
  /// all configurations, so a region with a clear preference gets its own hint.
  template <typename Fn> double costIn(unsigned St, Fn F) const {
    if (St != U)
      return F(St);
    double M = 0;
    for (unsigned C = 0; C < K; ++C)
      M = std::max(M, F(C));
    return M;
  }
  /// @brief Summary of the direct callee of CB, or nullptr if not (yet) solved.
  const FuncSummary *summary(const CallBase *CB) const {
    const Function *F = CB->getCalledFunction();
    auto It = Sums.find(F);
    return It == Sums.end() || !It->second.Done ? nullptr : &It->second;
  }
  /// @brief Cost of a call executed in state St: the callee's hinted cost if
  ///        it has hints, its uniform cost otherwise, or 50 instructions at
  ///        W* = 1 if it has no summary.
  double callUniform(const CallBase *CB, unsigned St) const {
    if (const FuncSummary *Sm = summary(CB)) {
      if (Sm->HasHints)
        return Sm->HintedCost;
      return costIn(St, [&](unsigned C) { return Sm->Uniform[C]; });
    }
    return costIn(St, [&](unsigned C) { return execCost(50, 1, C, TM, EW); });
  }
  /// @brief Dynamic instructions per call (50 if the callee has no summary).
  double callInsts(const CallBase *CB) const {
    const FuncSummary *Sm = summary(CB);
    return Sm ? Sm->Insts : 50;
  }

  /// @brief Body items of N: own instructions first, then child loops and
  ///        calls ordered by RPO position (memoized; references stay valid).
  const std::vector<Item> &items(LoopDemand *N);
  /// @brief Top-level items of a function: straight-line code, then top-level
  ///        loops and straight-line calls in RPO order.
  std::vector<Item> topItems(FunctionDemand &FD);
  /// @brief Sequence DP over Items (one execution, not multiplied by a trip count).
  /// @param Items      Items in program order.
  /// @param Start      State before the first item.
  /// @param RequireEnd If >= 0, the state after the last item must equal it.
  /// @param NeedSet    If true, some item must have set the state (hint, or a
  ///                   child/callee that changes it).
  /// @return Best result with per-item choices; Cost is INFINITY if infeasible.
  GRes seq(const std::vector<Item> &Items, unsigned Start, int RequireEnd, bool NeedSet);
  /// @brief G(N, St): cheapest execution of one entry of N entered in state St,
  ///        enumerating the steady state at the top of each iteration (memoized).
  const GRes &solveLoop(LoopDemand *N, unsigned St);
  /// @brief Walk the chosen solution R of Items recursively and append the
  ///        setwin decisions of child loops to Out.
  void reconstruct(const std::vector<Item> &Items, const GRes &R, std::vector<Decision> &Out);
};

// Placer members are documented at their declarations above.
double Placer::instsWithCalls(LoopDemand *N) {
  auto It = InstsMemo.find(N);
  if (It != InstsMemo.end())
    return It->second;
  double Body = N->OwnInsts;
  for (LoopDemand *C : N->Children)
    Body += instsWithCalls(C);
  for (auto &[CB, Pos] : N->Calls)
    Body += callInsts(CB);
  double R = N->Trip * Body;
  InstsMemo[N] = R;
  return R;
}

double Placer::uniformLoop(LoopDemand *N, unsigned C) {
  auto Key = std::make_pair(N, C);
  auto It = UniMemo.find(Key);
  if (It != UniMemo.end())
    return It->second;
  double Body = execCost(N->OwnInsts, N->WStar, C, TM, EW);
  for (LoopDemand *Ch : N->Children)
    Body += uniformLoop(Ch, C);
  for (auto &[CB, Pos] : N->Calls)
    Body += callUniform(CB, C);
  double R = N->Trip * Body;
  UniMemo[Key] = R;
  return R;
}

const std::vector<Item> &Placer::items(LoopDemand *N) {
  auto It = ItemsMemo.find(N);
  if (It != ItemsMemo.end())
    return It->second;
  std::vector<Item> V;
  Item Own;
  Own.K = Item::Own;
  Own.Insts = N->OwnInsts;
  Own.W = N->WStar;
  V.push_back(Own);
  std::vector<Item> Rest;
  for (LoopDemand *C : N->Children) {
    Item I;
    I.K = Item::LoopK;
    I.L = C;
    I.Pos = C->RPOIndex;
    Rest.push_back(I);
  }
  for (auto &[CB, Pos] : N->Calls) {
    Item I;
    I.K = Item::CallK;
    I.CB = CB;
    I.Pos = Pos;
    Rest.push_back(I);
  }
  std::stable_sort(Rest.begin(), Rest.end(),
                   [](const Item &A, const Item &B) { return A.Pos < B.Pos; });
  V.insert(V.end(), Rest.begin(), Rest.end());
  return ItemsMemo[N] = V;
}

std::vector<Item> Placer::topItems(FunctionDemand &FD) {
  std::vector<Item> V;
  Item Own;
  Own.K = Item::Own;
  Own.Insts = FD.StraightInsts;
  Own.W = 1;
  V.push_back(Own);
  std::vector<Item> Rest;
  for (LoopDemand *C : FD.TopLevel) {
    Item I;
    I.K = Item::LoopK;
    I.L = C;
    I.Pos = C->RPOIndex;
    Rest.push_back(I);
  }
  for (auto &[CB, Pos] : FD.StraightCalls) {
    Item I;
    I.K = Item::CallK;
    I.CB = CB;
    I.Pos = Pos;
    Rest.push_back(I);
  }
  std::stable_sort(Rest.begin(), Rest.end(),
                   [](const Item &A, const Item &B) { return A.Pos < B.Pos; });
  V.insert(V.end(), Rest.begin(), Rest.end());
  return V;
}

Placer::GRes Placer::seq(const std::vector<Item> &Items, unsigned Start, int RequireEnd,
                         bool NeedSet) {
  const unsigned NS = K + 1; // states incl. U
  const size_t N = Items.size();
  // dp[i][state][set]
  struct Cell {
    double Cost = INFINITY;
    int PrevSt = -1, PrevSet = -1, Choice = -1, Entry = -1;
  };
  std::vector<std::vector<std::array<Cell, 2>>> DP(N + 1, std::vector<std::array<Cell, 2>>(NS));
  DP[0][Start][0].Cost = 0;
  for (size_t I = 0; I < N; ++I) {
    const Item &It = Items[I];
    for (unsigned St = 0; St < NS; ++St)
      for (unsigned Set = 0; Set < 2; ++Set) {
        double Base = DP[I][St][Set].Cost;
        if (std::isinf(Base))
          continue;
        auto Relax = [&](unsigned NSt, unsigned NSet, double C, int Choice, int Entry) {
          Cell &T = DP[I + 1][NSt][NSet];
          if (Base + C < T.Cost - 1e-9) {
            T.Cost = Base + C;
            T.PrevSt = St;
            T.PrevSet = Set;
            T.Choice = Choice;
            T.Entry = Entry;
          }
        };
        switch (It.K) {
        case Item::Own:
          Relax(St, Set,
                costIn(St, [&](unsigned C) { return execCost(It.Insts, It.W, C, TM, EW); }), -1,
                St);
          break;
        case Item::CallK: {
          const FuncSummary *Sm = summary(It.CB);
          if (Sm && Sm->HasHints) {
            unsigned NSt = Sm->Exit >= 0 ? (unsigned)Sm->Exit : U;
            Relax(NSt, 1, Sm->HintedCost, -1, St);
          } else
            Relax(St, Set, callUniform(It.CB, St), -1, St);
          break;
        }
        case Item::LoopK: {
          const GRes &NoHint = solveLoop(It.L, St);
          bool ChildSets = NoHint.Exit != (int)St;
          Relax(NoHint.Exit, Set | ChildSets, NoHint.Cost, -1, St);
          if (instsWithCalls(It.L) >= MinInsts)
            for (unsigned C = 0; C < K; ++C) {
              if (C == St)
                continue;
              const GRes &G = solveLoop(It.L, C);
              Relax(G.Exit, 1, H + Seff + G.Cost, (int)C, (int)C);
            }
          break;
        }
        }
      }
  }
  GRes R;
  int BestSt = -1, BestSet = -1;
  for (unsigned St = 0; St < NS; ++St)
    for (unsigned Set = 0; Set < 2; ++Set) {
      if (RequireEnd >= 0 && (int)St != RequireEnd)
        continue;
      if (NeedSet && !Set)
        continue;
      if (DP[N][St][Set].Cost < R.Cost - 1e-9) {
        R.Cost = DP[N][St][Set].Cost;
        BestSt = St;
        BestSet = Set;
      }
    }
  if (BestSt < 0)
    return R; // infeasible
  R.Exit = BestSt;
  R.Choice.assign(N, -1);
  R.EntryState.assign(N, -1);
  int St = BestSt, Set = BestSet;
  for (size_t I = N; I > 0; --I) {
    const Cell &C = DP[I][St][Set];
    R.Choice[I - 1] = C.Choice;
    R.EntryState[I - 1] = C.Entry;
    St = C.PrevSt;
    Set = C.PrevSet;
  }
  return R;
}

const Placer::GRes &Placer::solveLoop(LoopDemand *N, unsigned St) {
  auto Key = std::make_pair(N, St);
  auto It = Memo.find(Key);
  if (It != Memo.end())
    return It->second;
  Memo[Key] = GRes(); // recursion guard (loop trees are acyclic)
  const std::vector<Item> &Items = items(N);
  GRes Best;
  for (unsigned E = 0; E <= K; ++E) {
    if (E == U && St != U)
      continue;
    GRes R = seq(Items, E, (int)E, /*NeedSet=*/E != St);
    if (std::isinf(R.Cost))
      continue;
    double Total = N->Trip * R.Cost + (E != St ? Seff : 0);
    if (Total < Best.Cost - 1e-9) {
      Best = R;
      Best.Cost = Total;
      Best.Exit = E;
    }
  }
  if (std::isinf(Best.Cost)) {
    // No steady state (e.g. a callee leaves the window unknown): relax.
    GRes R = seq(Items, St, -1, false);
    Best = R;
    Best.Cost = N->Trip * R.Cost;
  }
  return Memo[Key] = Best;
}

void Placer::reconstruct(const std::vector<Item> &Items, const GRes &R,
                         std::vector<Decision> &Out) {
  for (size_t I = 0; I < Items.size(); ++I) {
    if (Items[I].K != Item::LoopK)
      continue;
    LoopDemand *C = Items[I].L;
    int Entry = R.EntryState[I];
    if (Entry < 0)
      continue;
    if (R.Choice[I] >= 0)
      Out.push_back({C, (unsigned)R.Choice[I]});
    const GRes &G = solveLoop(C, (unsigned)Entry);
    reconstruct(items(C), G, Out);
  }
}

void Placer::solveFunction(FunctionDemand &FD, std::vector<Decision> &Out) {
  FuncSummary &Sm = Sums[FD.F];
  std::vector<Item> Top = topItems(FD);
  // Uniform costs and work.
  Sm.Uniform.assign(K, 0);
  double Insts = FD.StraightInsts, WNum = 0, WDen = 0;
  for (unsigned C = 0; C < K; ++C)
    Sm.Uniform[C] = execCost(FD.StraightInsts, 1, C, TM, EW);
  for (LoopDemand *L : FD.TopLevel) {
    Insts += instsWithCalls(L);
    WNum += L->DynInsts * L->NestWStar;
    WDen += L->DynInsts;
    for (unsigned C = 0; C < K; ++C)
      Sm.Uniform[C] += uniformLoop(L, C);
  }
  for (auto &[CB, Pos] : FD.StraightCalls) {
    Insts += callInsts(CB);
    for (unsigned C = 0; C < K; ++C)
      Sm.Uniform[C] += callUniform(CB, C);
  }
  Sm.Insts = std::max(1.0, Insts);
  Sm.DemandW = WDen > 0 ? WNum / WDen : 0;

  GRes R = seq(Top, U, -1, false);
  size_t Before = Out.size();
  reconstruct(Top, R, Out);
  Sm.HasHints = Out.size() > Before || R.Exit != (int)U;
  Sm.HintedCost = R.Cost;
  Sm.Exit = R.Exit == (int)U ? -1 : R.Exit;
  Sm.Done = true;
}

//===----------------------------------------------------------------------===//
// Region numbering and JSON I/O.
//===----------------------------------------------------------------------===//

/// A region: one top-level loop nest.
struct RegionInfo {
  unsigned Id;      ///< logical id (may exceed 63)
  unsigned Encoded; ///< id carried by the hint (min(Id, 63))
  Function *F;      ///< containing function
  LoopDemand *D;    ///< top-level loop
};

/// @brief Kernel name for output files: -winhint-kernel, else the source file
///        (or module identifier) stem up to its first '.'.
std::string kernelName(Module &M) {
  if (!OptKernel.empty())
    return OptKernel;
  StringRef Src = M.getSourceFileName();
  if (Src.empty())
    Src = M.getModuleIdentifier();
  std::string Stem = sys::path::stem(Src).str();
  // a.b.c -> a (e.g. kernel.clairvoyance.bc)
  return StringRef(Stem).split('.').first.str();
}

/**
 * @brief Load a region -> W map for -winhint-mode=from-json.
 *
 * The root (or its "regions" member) is either an object keyed by decimal
 * region id or an array of objects with "region"/"id". A value may be an
 * integer config index (negative = release), the string "release", or an
 * object with "W"/"w"/"rob"/"window" (entries) or "config"/"best_config"
 * (index). Config indices are clamped to the table and mapped to their ROB.
 * Entries that do not match are skipped.
 *
 * @param[in]  Path JSON file.
 * @param[in]  TM   Target model (config index -> ROB).
 * @param[out] Map  Region id -> W in entries (0 = release).
 * @param[out] Err  Error message on failure.
 * @return false if the file cannot be read or parsed, or the root is neither
 *         an object nor an array.
 */
bool loadRegionMap(StringRef Path, const TargetModel &TM, std::map<unsigned, unsigned> &Map,
                   std::string &Err) {
  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf) {
    Err = "cannot read " + Path.str();
    return false;
  }
  Expected<json::Value> V = json::parse((*Buf)->getBuffer());
  if (!V) {
    Err = toString(V.takeError());
    return false;
  }
  auto ToW = [&](const json::Value &E, unsigned &W) -> bool {
    if (auto I = E.getAsInteger()) {
      if (*I < 0) {
        W = 0;
        return true;
      }
      W = TM.Window[std::min<size_t>(*I, TM.Window.size() - 1)].ROB;
      return true;
    }
    if (auto S = E.getAsString())
      if (*S == "release") {
        W = 0;
        return true;
      }
    if (const json::Object *O = E.getAsObject()) {
      for (const char *Key : {"W", "w", "rob", "window"})
        if (auto X = O->getInteger(Key)) {
          W = *X < 0 ? 0 : (unsigned)std::min<int64_t>(*X, UINT32_MAX); // < 0: release
          return true;
        }
      for (const char *Key : {"config", "best_config"})
        if (auto X = O->getInteger(Key)) {
          W = *X < 0 ? 0 : TM.Window[std::min<size_t>(*X, TM.Window.size() - 1)].ROB;
          return true;
        }
    }
    return false;
  };
  const json::Value *Root = &*V;
  if (const json::Object *O = Root->getAsObject())
    if (const json::Value *R = O->get("regions"))
      Root = R;
  if (const json::Object *O = Root->getAsObject()) {
    for (auto &KV : *O) {
      unsigned Id;
      if (!to_integer(StringRef(KV.first), Id, 10))
        continue;
      unsigned W;
      if (ToW(KV.second, W))
        Map[Id] = W;
    }
  } else if (const json::Array *A = Root->getAsArray()) {
    for (const json::Value &E : *A) {
      const json::Object *O = E.getAsObject();
      if (!O)
        continue;
      auto Id = O->getInteger("region");
      if (!Id)
        Id = O->getInteger("id");
      unsigned W;
      if (Id && ToW(E, W))
        Map[(unsigned)*Id] = W;
    }
  } else {
    Err = "unrecognized region map format";
    return false;
  }
  return true;
}

/// @brief Write Dir/Name (2-space indented JSON produced by Body), creating
///        Dir; does nothing if Dir is empty, warns if the file cannot be opened.
void writeJSONFile(StringRef Dir, StringRef Name, function_ref<void(json::OStream &)> Body) {
  if (Dir.empty())
    return;
  sys::fs::create_directories(Dir);
  SmallString<256> P(Dir);
  sys::path::append(P, Name);
  std::error_code EC;
  raw_fd_ostream OS(P, EC, sys::fs::OF_Text);
  if (EC) {
    errs() << "winhint: warning: cannot write " << P << ": " << EC.message() << "\n";
    return;
  }
  json::OStream J(OS, 2);
  Body(J);
  OS << "\n";
}

/// @brief Return the preheader of L, inserting one if needed (may return
///        nullptr); sets Changed when a block is inserted.
BasicBlock *ensurePreheader(Loop *L, DominatorTree &DT, LoopInfo &LI, bool &Changed) {
  if (BasicBlock *PH = L->getLoopPreheader())
    return PH;
  BasicBlock *PH = InsertPreheaderForLoop(L, &DT, &LI, nullptr, /*PreserveLCSSA=*/false);
  Changed |= PH != nullptr;
  return PH;
}

} // namespace

//===----------------------------------------------------------------------===//
// The pass.
//===----------------------------------------------------------------------===//

// Documented in HintPlacement.h.
PreservedAnalyses HintPlacementPass::run(Module &M, ModuleAnalysisManager &MAM) {
  auto T0 = std::chrono::steady_clock::now();
  Mode Md = Mode::Model;
  std::string MapPath;
  StringRef ModeStr = OptMode;
  if (ModeStr == "setwin" || ModeStr == "model" || ModeStr.empty())
    Md = Mode::Model;
  else if (ModeStr == "regions")
    Md = Mode::Regions;
  else if (ModeStr.starts_with("from-json=")) {
    Md = Mode::FromJSON;
    MapPath = ModeStr.drop_front(strlen("from-json=")).str();
  } else if (ModeStr == "off")
    return PreservedAnalyses::all();
  else
    reportFatalUsageError("winhint: unknown -winhint-mode=" + ModeStr);

  EmitMode EM;
  if (!parseEmitMode(OptEmit, EM))
    reportFatalUsageError("winhint: unknown -winhint-emit=" + StringRef(OptEmit));

  const TargetModel &TM = getTargetModel();
  if (OptVerbose) {
    errs() << "winhint: ";
    TM.print(errs());
  }
  HintEmitter Emitter(M, EM);
  if (EM == EmitMode::Asm && !Emitter.asmSupported()) {
    errs() << "winhint: warning: no asm hint encoding for triple '" << M.getTargetTriple().str()
           << "'; hints are not emitted\n";
  }

  // Switch cost.
  double S = OptSwitchCost;
  if (S <= 0)
    S = StringRef(OptSwitchModel) == "pe" ? OptMigrationUs * 1000.0 * TM.ClockGHz
                                         : TM.gem5SwitchCost();
  double H = OptHintCost > 0 ? (double)OptHintCost : (EM == EmitMode::Call ? 8.0 : 1.0);
  double MinInsts = OptMinRegionInsts > 0 ? (double)OptMinRegionInsts : 2.0 * TM.wMax();

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Functions sorted by name (region numbering, docs/interfaces.md §5).
  std::vector<Function *> Funcs;
  for (Function &F : M)
    if (!F.isDeclaration())
      Funcs.push_back(&F);
  std::stable_sort(Funcs.begin(), Funcs.end(),
                   [](Function *A, Function *B) { return A->getName() < B->getName(); });

  DenseMap<Function *, FunctionDemand *> Demand;
  for (Function *F : Funcs)
    Demand[F] = &FAM.getResult<WindowDemandAnalysis>(*F);
  double AnalysisMs =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();

  std::vector<RegionInfo> Regions;
  DenseMap<LoopDemand *, unsigned> RegionOf;
  for (Function *F : Funcs)
    for (LoopDemand *D : Demand[F]->TopLevel) {
      unsigned Id = Regions.size();
      Regions.push_back({Id, std::min(Id, 63u), F, D});
      RegionOf[D] = Id;
    }
  if (Regions.size() > 64 && (Md != Mode::Model || OptRegionMarkers))
    errs() << "winhint: warning: " << Regions.size()
           << " regions; ids >= 63 share the overflow marker region(63)\n";

  // ---- Decide -------------------------------------------------------------
  DenseMap<Function *, std::vector<Decision>> Decisions;
  DenseMap<const Function *, FuncSummary> Sums;
  std::map<unsigned, unsigned> RegionMap;
  if (Md == Mode::FromJSON) {
    std::string Err;
    if (!loadRegionMap(MapPath, TM, RegionMap, Err))
      reportFatalUsageError("winhint: -winhint-mode=from-json: " + StringRef(Err));
  }
  if (Md == Mode::Model) {
    Placer P(TM, Sums, S, OptHysteresis, H, OptEnergyWeight, MinInsts);
    // Bottom-up over the call graph (callees first).
    DenseMap<Function *, int> State; // 0 new, 1 active, 2 done
    std::vector<Function *> PostOrder;
    std::function<void(Function *)> Visit = [&](Function *F) {
      State[F] = 1;
      FunctionDemand *FD = Demand[F];
      std::vector<const CallBase *> Calls;
      for (auto &[CB, Pos] : FD->StraightCalls)
        Calls.push_back(CB);
      for (auto &DP : FD->Loops)
        for (auto &[CB, Pos] : DP->Calls)
          Calls.push_back(CB);
      for (const CallBase *CB : Calls) {
        Function *Callee = CB->getCalledFunction();
        if (Callee && Demand.count(Callee) && State.lookup(Callee) == 0)
          Visit(Callee);
      }
      State[F] = 2;
      PostOrder.push_back(F);
    };
    for (Function *F : Funcs)
      if (State.lookup(F) == 0)
        Visit(F);
    for (Function *F : PostOrder)
      P.solveFunction(*Demand[F], Decisions[F]);
  }

  auto RegionOfLoop = [&](LoopDemand *D) -> int {
    while (D->Parent)
      D = D->Parent;
    auto It = RegionOf.find(D);
    return It == RegionOf.end() ? -1 : (int)It->second;
  };
  // Per region: setwin values emitted inside the nest, and the value at the
  // nest entry (hint on the top-level loop itself), -1 = inherited.
  std::map<unsigned, std::vector<unsigned>> RegionSetWins;
  std::map<unsigned, int> RegionEntryW;

  // ---- Emit -----------------------------------------------------------------
  // Hints actually inserted into the IR (Emitter.emit() returns nullptr for
  // -winhint-emit=none and for asm mode on an unsupported triple); the
  // `hints` log below still lists every placed hint.
  unsigned NumSetWin = 0, NumRegion = 0;
  bool CFGChanged = false; // preheaders inserted
  json::Array HintLog;
  bool Markers = Md == Mode::Regions || Md == Mode::FromJSON || OptRegionMarkers;
  for (Function *F : Funcs) {
    FunctionDemand *FD = Demand[F];
    auto &LI = FAM.getResult<LoopAnalysis>(*F);
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
    OptimizationRemarkEmitter ORE(F);
    struct Pending {
      LoopDemand *D;
      HintKind K;
      unsigned V;
    };
    std::vector<Pending> Todo;
    if (Markers)
      for (LoopDemand *D : FD->TopLevel)
        Todo.push_back({D, HintKind::Region, std::min(RegionOf[D], 63u)});
    if (Md == Mode::FromJSON)
      for (LoopDemand *D : FD->TopLevel) {
        // Maps keyed by the id seen at run time (gem5 region_stats.csv) only
        // know the overflow marker 63 for every id >= 63.
        auto It = RegionMap.find(RegionOf[D]);
        if (It == RegionMap.end() && RegionOf[D] > 63)
          It = RegionMap.find(63);
        if (It != RegionMap.end())
          Todo.push_back({D, HintKind::SetWin, It->second});
      }
    if (Md == Mode::Model)
      for (Decision &Dc : Decisions[F])
        Todo.push_back({Dc.D, HintKind::SetWin, TM.Window[Dc.Config].ROB});
    for (Pending &P : Todo) {
      BasicBlock *PH = ensurePreheader(P.D->L, DT, LI, CFGChanged);
      if (!PH) {
        errs() << "winhint: warning: no preheader for loop in " << F->getName() << "\n";
        continue;
      }
      Instruction *Hint = Emitter.emit(PH->getTerminator(), P.K, P.V);
      if (Hint) {
        Hint->setDebugLoc(P.D->L->getStartLoc());
        (P.K == HintKind::SetWin ? NumSetWin : NumRegion)++;
      }
      int Rg = RegionOfLoop(P.D);
      if (P.K == HintKind::SetWin && Rg >= 0) {
        RegionSetWins[Rg].push_back(P.V);
        if (!P.D->Parent)
          RegionEntryW[Rg] = (int)P.V;
      }
      json::Object E{{"function", F->getName()},
                     {"region", (int64_t)Rg},
                     {"line", (int64_t)P.D->Line},
                     {"loop", P.D->HeaderName},
                     {"depth", (int64_t)P.D->Depth},
                     {"kind", P.K == HintKind::SetWin ? "setwin" : "region"},
                     {"value", (int64_t)P.V},
                     {"emitted", Hint != nullptr},
                     {"encoding", Emitter.describe(P.K, P.V)}};
      HintLog.push_back(std::move(E));
      if (OptVerbose)
        errs() << "winhint: " << F->getName() << " line " << P.D->Line << " loop "
               << P.D->HeaderName << ": " << Emitter.describe(P.K, P.V) << "\n";
      ORE.emit([&]() {
        return OptimizationRemark("winhint", P.K == HintKind::SetWin ? "SetWin" : "Region",
                                  P.D->L->getStartLoc(), P.D->L->getHeader())
               << (P.K == HintKind::SetWin ? "setwin(" : "region(") << ore::NV("Value", P.V)
               << ") W*=" << ore::NV("WStar", P.D->WStar);
      });
    }
  }

  double Ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0)
                  .count();

  // ---- Outputs ----------------------------------------------------------------
  std::string Kernel = kernelName(M);
  auto RegionJSON = [&](json::OStream &J, const RegionInfo &R) {
    LoopDemand *D = R.D;
    J.attribute("function", R.F->getName());
    J.attribute("header", D->HeaderName);
    J.attribute("line", (int64_t)D->Line);
    J.attribute("encoded_id", (int64_t)R.Encoded);
    J.attribute("w_star", (int64_t)D->WStar);
    J.attribute("nest_w_star", std::round(D->NestWStar * 10) / 10);
    J.attribute("config", (int64_t)D->Config);
    J.attribute("nest_config", (int64_t)D->NestConfig);
    J.attribute("L_mem", std::round(D->Lmem * 10) / 10);
    J.attribute("D_indep", std::isinf(D->DIndep) ? json::Value(nullptr)
                                                 : json::Value(std::round(D->DIndep * 10) / 10));
    J.attribute("CP", std::round(D->CP * 10) / 10);
    J.attribute("footprint_bytes",
                std::isinf(D->NestFootprint) ? json::Value(nullptr) : json::Value(D->NestFootprint));
    J.attribute("dyn_insts_est", D->DynInsts);
    J.attribute("conservative", D->Conservative);
  };
  writeJSONFile(OptOutDir, Kernel + ".regions.json", [&](json::OStream &J) {
    J.object([&] {
      J.attribute("kernel", Kernel);
      J.attribute("target", TM.Name);
      J.attribute("num_regions", (int64_t)Regions.size());
      J.attribute("overflow", Regions.size() > 64);
      J.attributeObject("regions", [&] {
        for (const RegionInfo &R : Regions)
          J.attributeObject(std::to_string(R.Id), [&] { RegionJSON(J, R); });
      });
    });
  });
  SmallString<256> StatsDir(OptOutDir), StatsName(Kernel + ".winhint.json");
  if (!OptStatsFile.empty()) {
    StatsDir = sys::path::parent_path(OptStatsFile);
    if (StatsDir.empty())
      StatsDir = ".";
    StatsName = sys::path::filename(OptStatsFile);
  }
  writeJSONFile(StatsDir, StatsName, [&](json::OStream &J) {
    J.object([&] {
      J.attribute("schema", "winhint-stats/1");
      J.attribute("kernel", Kernel);
      J.attribute("module", M.getModuleIdentifier());
      J.attribute("llvm_version", LLVM_VERSION_STRING);
      J.attribute("mode", ModeStr);
      J.attribute("emit", emitModeName(EM));
      J.attribute("triple", M.getTargetTriple().str());
      J.attribute("target", TM.Name);
      J.attribute("switch_cost_cycles", S);
      J.attribute("hysteresis", (double)OptHysteresis);
      J.attribute("cp_model", StringRef(OptCPModel));
      J.attribute("min_region_insts", MinInsts);
      J.attribute("hints_setwin", (int64_t)NumSetWin);
      J.attribute("hints_region", (int64_t)NumRegion);
      J.attribute("compile_time_ms", Ms);
      J.attribute("analysis_time_ms", AnalysisMs);
      J.attribute("num_regions", (int64_t)Regions.size());
      J.attributeArray("window", [&] {
        for (const WindowConfig &C : TM.Window)
          J.object([&] {
            J.attribute("rob", (int64_t)C.ROB);
            J.attribute("iq", (int64_t)C.IQ);
            J.attribute("lq", (int64_t)C.LQ);
            J.attribute("sq", (int64_t)C.SQ);
          });
      });
      J.attributeArray("regions", [&] {
        for (const RegionInfo &R : Regions)
          J.object([&] {
            J.attribute("id", (int64_t)R.Id);
            RegionJSON(J, R);
            auto EIt = RegionEntryW.find(R.Id);
            if (EIt != RegionEntryW.end()) {
              J.attribute("entry_setwin", (int64_t)EIt->second);
              J.attribute("entry_config", (int64_t)TM.configForW(EIt->second));
            } else {
              J.attribute("entry_setwin", nullptr);
              J.attribute("entry_config", nullptr);
            }
            J.attributeArray("setwin", [&] {
              auto It = RegionSetWins.find(R.Id);
              if (It != RegionSetWins.end())
                for (unsigned V : It->second)
                  J.value((int64_t)V);
            });
          });
      });
      J.attributeArray("hints", [&] {
        for (json::Value &V : HintLog)
          J.value(V);
      });
      J.attributeArray("loops", [&] {
        for (Function *F : Funcs)
          for (auto &DP : Demand[F]->Loops) {
            LoopDemand *D = DP.get();
            J.object([&] {
              J.attribute("function", F->getName());
              J.attribute("header", D->HeaderName);
              J.attribute("line", (int64_t)D->Line);
              J.attribute("depth", (int64_t)D->Depth);
              auto It = RegionOf.find(D);
              if (It != RegionOf.end())
                J.attribute("region", (int64_t)It->second);
              J.attribute("trip", D->Trip);
              J.attribute("trip_known", D->TripKnown);
              J.attribute("body_insts", D->OwnInsts);
              J.attribute("L_mem", D->Lmem);
              J.attribute("misses_per_iter", D->MissesPerIter);
              J.attribute("D_indep", std::isinf(D->DIndep) ? json::Value(nullptr)
                                                           : json::Value(D->DIndep));
              J.attribute("MLP", D->MLPTarget);
              J.attribute("CP", D->CP);
              J.attribute("RecMII", D->RecMII);
              J.attribute("W_mlp", (int64_t)D->Wmlp);
              J.attribute("W_cp", (int64_t)D->Wcp);
              J.attribute("w_star", (int64_t)D->WStar);
              J.attribute("config", (int64_t)D->Config);
              J.attribute("conservative", D->Conservative);
            });
          }
      });
    });
  });
  if (OptVerbose)
    errs() << "winhint: " << Kernel << ": " << NumSetWin << " setwin, " << NumRegion
           << " region hints, " << Regions.size() << " regions, " << format("%.2f", Ms)
           << " ms\n";

  if (NumSetWin + NumRegion == 0 && !CFGChanged)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

} // namespace winhint
