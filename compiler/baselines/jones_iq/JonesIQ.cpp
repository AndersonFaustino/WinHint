//===- JonesIQ.cpp - B6: software-directed issue-queue resizing -----------===//
//
// Reimplementation of
//   T. M. Jones, M. F. P. O'Boyle, J. Abella, A. González,
//   "Software Directed Issue Queue Power Reduction", HPCA 2005
// (and the extended "Compiler Directed Issue Queue Energy Reduction",
//  Trans. HiPEAC 4(1), 2009), as an LLVM (23) new-PM plugin.
//
// For every region (each innermost loop, and every other basic block with at
// least -jones-min-block instructions) the pass builds the data-dependence
// DAG of the region's instructions in program order (loops: the body is
// replicated so that cross-iteration overlap is visible), schedules it on an
// idealized out-of-order machine (in-order dispatch of D per cycle, issue of
// up to IW ready instructions per cycle, per-op latencies, loads hit in L1 as
// in the paper) and finds the smallest issue-queue size (in banks of 8) that
// does not lengthen the schedule. A hint is placed at the region entry (loop
// preheader / block start); hints equal to the setting already in force on
// every path are removed (forward must-dataflow), as in the paper.
//
//   -jones-mode=iq    : IQ demand only, as published.
//   -jones-mode=full  : extension: ROB, IQ, LQ and SQ demand jointly; the
//                       smallest window configuration that does not lengthen
//                       the schedule.
//   -jones-iq-encoding=setwin (default) : IQ demand is encoded with the
//                       contract's setwin: W = ROB of the smallest config
//                       whose IQ covers the demand (whole window scales).
//   -jones-iq-encoding=setiq : PROPOSED (not in docs/interfaces.md) IQ-only
//                       hint: RISC-V tag 25, x86 kind 3, payload = IQ/8.
//
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief B6 baseline plugin (JonesIQ.so): Jones et al. software-directed
 *        issue-queue resizing, emitting hints through the WinHint HintEmitter.
 *
 * Registers the module pass `jones-iq` and, unless -jones-auto=false, adds it
 * at the optimizer-last extension point (skipped in LTO pre-link). With
 * -jones-out-dir it writes `<kernel>.jones.json` (or `.jones_full.json` in
 * full mode) with per-region demands and hint counts.
 */
#include "HintEmitter.h"
#include "TargetModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Plugins/PassPlugin.h") // LLVM >= 23
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <algorithm>
#include <chrono>

using namespace llvm;
using namespace winhint;

/// -jones-target: machine JSON (empty = builtin default machine).
static cl::opt<std::string> JTarget("jones-target", cl::desc("Machine JSON"), cl::init(""));
/// -jones-mode=iq|full: IQ demand only (as published) or joint ROB/IQ/LQ/SQ.
static cl::opt<std::string> JMode("jones-mode", cl::desc("iq (as published) | full (ROB+IQ+LSQ)"),
                                  cl::init("iq"));
/// -jones-iq-encoding=setwin|setiq: hint used in iq mode (full mode always uses setwin).
static cl::opt<std::string> JEnc("jones-iq-encoding",
                                 cl::desc("setwin (contract) | setiq (proposed tag)"),
                                 cl::init("setwin"));
/// -jones-emit=asm|call|none: hint materialization (see EmitMode).
static cl::opt<std::string> JEmit("jones-emit", cl::desc("asm | call | none"), cl::init("asm"));
/// -jones-min-block: minimum queue-occupying instructions for a block region (default 8).
static cl::opt<unsigned> JMinBlock("jones-min-block",
                                   cl::desc("Minimum instructions for a block region"),
                                   cl::init(8));
/// -jones-loop-insts: replicate loop bodies up to this many instructions (default 96).
static cl::opt<unsigned> JCopies("jones-loop-insts",
                                 cl::desc("Replicate loop bodies up to this many instructions"),
                                 cl::init(96));
/// -jones-tolerance: allowed schedule lengthening as a fraction (default 0).
static cl::opt<double> JTolerance("jones-tolerance",
                                  cl::desc("Allowed schedule lengthening (fraction)"),
                                  cl::init(0.0));
/// -jones-out-dir: directory for the per-kernel JSON log (empty = none).
static cl::opt<std::string> JOutDir("jones-out-dir", cl::desc("Directory for <kernel>.jones.json"),
                                    cl::init(""));
/// -jones-kernel: kernel name for the output file (default: source stem).
static cl::opt<std::string> JKernel("jones-kernel", cl::desc("Kernel name"), cl::init(""));
/// -jones-auto: add the pass at the optimizer-last extension point (default on).
static cl::opt<bool> JAuto("jones-auto", cl::desc("Run at the optimizer-last EP"), cl::init(true));
/// -jones-verbose: print each region and its hint to stderr.
static cl::opt<bool> JVerbose("jones-verbose", cl::desc("Print regions"), cl::init(false));

namespace {

/// @brief Target model from -jones-target, through the shared thread-safe
///        per-path cache getCachedTargetModel() (builtin default, with a
///        warning, if loading fails; fatal error on an empty window table).
const TargetModel &target() { return getCachedTargetModel(JTarget, "jones-iq"); }

/// @brief Latency of an instruction in cycles (L1 hits, as in the paper).
///
/// Intrinsic calls other than sqrt/fma/fmuladd use the FP-add latency if their
/// result or an argument is floating point (scalar or vector) and the integer
/// ALU latency otherwise (integer intrinsics); other calls use
/// OpLatencies::Call; unlisted opcodes the integer ALU latency.
unsigned latency(const Instruction &I, const TargetModel &TM) {
  const OpLatencies &L = TM.Lat;
  switch (I.getOpcode()) {
  case Instruction::Mul:
    return L.IntMul;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
    return L.IntDiv;
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FNeg:
  case Instruction::FCmp:
    return L.FpAdd;
  case Instruction::FMul:
    return L.FpMul;
  case Instruction::FDiv:
  case Instruction::FRem:
    return L.FpDiv;
  case Instruction::FPToSI:
  case Instruction::FPToUI:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
    return L.FpCvt;
  case Instruction::Load:
    return TM.l1Latency();
  case Instruction::Store:
    return L.Store;
  case Instruction::Call: {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::sqrt:
        return L.FpSqrt;
      case Intrinsic::fma:
      case Intrinsic::fmuladd:
        return L.FpFma;
      default: {
        // Integer intrinsics (smax, ctpop, bswap, memcpy, ...) run on the
        // integer ALU; anything with an FP result or operand on the FP adder.
        bool FP = II->getType()->isFPOrFPVectorTy();
        for (const Value *A : II->args())
          FP |= A->getType()->isFPOrFPVectorTy();
        return FP ? L.FpAdd : L.IntAlu;
      }
      }
    }
    return L.Call;
  }
  default:
    return L.IntAlu;
  }
}

/// @brief Whether I occupies a queue entry (not PHIs, debug/pseudo or
///        lifetime intrinsics, bitcasts, freeze, constant-index GEPs or
///        unconditional branches).
bool occupies(const Instruction &I) {
  if (isa<PHINode>(I) || I.isDebugOrPseudoInst() || I.isLifetimeStartOrEnd())
    return false;
  if (isa<BitCastInst>(I) || isa<FreezeInst>(I))
    return false;
  if (auto *G = dyn_cast<GetElementPtrInst>(&I))
    return !G->hasAllConstantIndices();
  // Unconditional branches occupy no entry (LLVM >= 23 has a separate
  // UncondBr opcode; older versions use one Br opcode with 1 or 3 operands).
#if LLVM_VERSION_MAJOR >= 23
  if (isa<UncondBrInst>(I))
    return false;
#else
  if (auto *B = dyn_cast<BranchInst>(&I))
    return B->isConditional();
#endif
  return true;
}

/// One instruction of a region sequence.
struct Node {
  unsigned Lat;         ///< execution latency (cycles)
  bool IsLoad, IsStore; ///< occupies an LQ / SQ entry
  SmallVector<int, 4> Preds; // indices of producers in the sequence
};

/// Structure capacities used by schedule().
struct Caps {
  unsigned ROB, IQ, LQ, SQ; ///< entries
};

/// @brief Schedule Seq on an idealized out-of-order core and return the
///        cycle of the last commit (0 for an empty sequence).
///
/// In-order dispatch of up to DispatchWidth per cycle, subject to the IQ/ROB/
/// LQ/SQ capacities in C; issue of up to IssueWidth ready instructions per
/// cycle; in-order commit of up to CommitWidth per cycle. O(n^2).
unsigned schedule(const std::vector<Node> &Seq, const Caps &C, const TargetModel &TM) {
  const unsigned DW = std::max(1u, TM.DispatchWidth), IW = std::max(1u, TM.IssueWidth),
                 CW = std::max(1u, TM.CommitWidth);
  size_t N = Seq.size();
  std::vector<unsigned> Disp(N), Iss(N), Fin(N), Com(N);
  DenseMap<unsigned, unsigned> IssUsed, ComUsed, DispUsed;
  unsigned LastDisp = 0, LastCom = 0;
  for (size_t I = 0; I < N; ++I) {
    const Node &Nd = Seq[I];
    // Dispatch: in order, DW per cycle, structure caps.
    unsigned T = LastDisp;
    for (;; ++T) {
      if (DispUsed.lookup(T) >= DW)
        continue;
      unsigned InIQ = 0, InROB = 0, InLQ = 0, InSQ = 0;
      for (size_t J = 0; J < I; ++J) {
        if (Iss[J] > T)
          ++InIQ;
        if (Com[J] > T) {
          ++InROB;
          InLQ += Seq[J].IsLoad;
          InSQ += Seq[J].IsStore;
        }
      }
      if (InIQ < C.IQ && InROB < C.ROB && (!Nd.IsLoad || InLQ < C.LQ) &&
          (!Nd.IsStore || InSQ < C.SQ))
        break;
    }
    Disp[I] = T;
    DispUsed[T]++;
    LastDisp = T;
    unsigned Ready = T + 1;
    for (int P : Nd.Preds)
      Ready = std::max(Ready, Fin[P]);
    unsigned IT = Ready;
    while (IssUsed.lookup(IT) >= IW)
      ++IT;
    IssUsed[IT]++;
    Iss[I] = IT;
    Fin[I] = IT + Nd.Lat;
    unsigned CT = std::max(Fin[I], LastCom);
    while (ComUsed.lookup(CT) >= CW)
      ++CT;
    ComUsed[CT]++;
    Com[I] = CT;
    LastCom = CT;
  }
  return N ? LastCom : 0;
}

/// @brief Build the region sequence. For loops the body (blocks in RPO) is
///        replicated; header PHIs link copy k to the latch value of copy k-1.
/// @param Blocks   Blocks of the region in order.
/// @param L        Loop for a loop region, nullptr for a single block.
/// @param TM       Target model (latencies).
/// @param MaxInsts Loop bodies get clamp(MaxInsts / body, 2, 16) copies.
/// @return Nodes with producer indices; truncated at 384 nodes.
std::vector<Node> buildSequence(ArrayRef<BasicBlock *> Blocks, Loop *L, const TargetModel &TM,
                                unsigned MaxInsts) {
  std::vector<Instruction *> Body;
  for (BasicBlock *BB : Blocks)
    for (Instruction &I : *BB)
      Body.push_back(&I);
  unsigned PerIter = 0;
  for (Instruction *I : Body)
    PerIter += occupies(*I);
  unsigned Copies = 1;
  if (L && PerIter)
    Copies = std::clamp(MaxInsts / std::max(1u, PerIter), 2u, 16u);
  std::vector<Node> Seq;
  DenseMap<const Value *, int> Prev, Cur; // value -> node index (current/previous copy)
  BasicBlock *Latch = L ? L->getLoopLatch() : nullptr;
  for (unsigned K = 0; K < Copies; ++K) {
    Cur.clear();
    for (Instruction *I : Body) {
      if (auto *P = dyn_cast<PHINode>(I)) {
        // Map the PHI to its producer in the previous copy.
        if (K > 0 && Latch && L && P->getParent() == L->getHeader() &&
            P->getBasicBlockIndex(Latch) >= 0) {
          auto It = Prev.find(P->getIncomingValueForBlock(Latch));
          if (It != Prev.end())
            Cur[P] = It->second;
        }
        continue;
      }
      if (!occupies(*I)) {
        // Free instruction: forward its operand's producer.
        if (I->getNumOperands())
          if (auto It = Cur.find(I->getOperand(0)); It != Cur.end())
            Cur[I] = It->second;
        continue;
      }
      Node Nd;
      Nd.Lat = latency(*I, TM);
      Nd.IsLoad = isa<LoadInst>(I);
      Nd.IsStore = isa<StoreInst>(I);
      for (Value *Op : I->operands()) {
        auto It = Cur.find(Op);
        if (It != Cur.end())
          Nd.Preds.push_back(It->second);
      }
      Cur[I] = Seq.size();
      Seq.push_back(Nd);
      if (Seq.size() >= 384) // bound the O(n^2) scheduler on huge blocks
        return Seq;
    }
    Prev = Cur;
  }
  return Seq;
}

/// Result of analyze() for one region.
struct Demand {
  unsigned IQ = 0;     ///< smallest IQ (multiple of 8) without slowdown
  unsigned Config = 0; ///< selected window config index
  unsigned Hint = 0; // hint value (W for setwin, IQ for setiq)
  unsigned Insts = 0;  ///< nodes in the scheduled sequence
};

/// @brief Find the region's IQ demand (and, in full mode, the smallest
///        window config) that keeps the schedule within the tolerance of the
///        largest configuration.
/// @param Seq   Region sequence from buildSequence().
/// @param TM    Target model (window table must not be empty).
/// @param Full  Joint ROB/IQ/LQ/SQ search instead of IQ only.
/// @param SetIQ Hint carries the IQ size (setiq) instead of a ROB size.
/// @return The demand and the hint value.
Demand analyze(const std::vector<Node> &Seq, const TargetModel &TM, bool Full, bool SetIQ) {
  Demand D;
  D.Insts = Seq.size();
  const WindowConfig &Big = TM.Window.back();
  Caps Max{Big.ROB, Big.IQ, Big.LQ, Big.SQ};
  unsigned Ref = schedule(Seq, Max, TM);
  unsigned Limit = Ref + (unsigned)(Ref * JTolerance);
  // Smallest IQ (banks of 8) that does not lengthen the schedule.
  unsigned Q = 8;
  for (; Q < Big.IQ; Q += 8) {
    Caps C = Max;
    C.IQ = Q;
    if (schedule(Seq, C, TM) <= Limit)
      break;
  }
  D.IQ = std::min(Q, Big.IQ);
  if (!Full) {
    D.Config = TM.configForIQ(D.IQ);
    D.Hint = SetIQ ? D.IQ : TM.Window[D.Config].ROB;
    return D;
  }
  // Extension: smallest whole-window configuration without slowdown.
  D.Config = TM.Window.size() - 1;
  for (unsigned C = 0; C < TM.Window.size(); ++C) {
    const WindowConfig &W = TM.Window[C];
    if (schedule(Seq, Caps{W.ROB, W.IQ, W.LQ, W.SQ}, TM) <= Limit) {
      D.Config = C;
      break;
    }
  }
  D.Hint = TM.Window[D.Config].ROB;
  return D;
}

/// @brief Kernel name: -jones-kernel, else the source file (or module
///        identifier) stem up to its first '.' (as WinHint's kernelName()).
std::string kernelName(Module &M) {
  if (!JKernel.empty())
    return JKernel;
  StringRef Src = M.getSourceFileName();
  if (Src.empty())
    Src = M.getModuleIdentifier();
  return StringRef(sys::path::stem(Src)).split('.').first.str();
}

/// B6 module pass `jones-iq`.
struct JonesIQPass : PassInfoMixin<JonesIQPass> {
  /// @brief Required pass: never skipped by optnone or opt-bisect.
  static bool isRequired() { return true; }

  /// @brief Analyze every innermost loop and every large-enough block outside
  ///        innermost loops, drop hints equal to the setting already in force
  ///        on all paths, emit the rest and write the JSON log.
  /// @return PreservedAnalyses::none() if hints were emitted or preheaders
  ///         were inserted, else all().
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    auto T0 = std::chrono::steady_clock::now();
    const TargetModel &TM = target();
    bool Full = JMode == "full";
    bool SetIQ = !Full && JEnc == "setiq";
    EmitMode EM;
    if (!parseEmitMode(JEmit, EM))
      reportFatalUsageError("jones-iq: bad -jones-emit=" + StringRef(JEmit) +
                            " (expected asm, call or none)");
    HintEmitter Emitter(M, EM);
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    HintKind Kind = SetIQ ? HintKind::SetIQ : HintKind::SetWin;

    struct Planned {
      BasicBlock *At;   // block whose entry state is set
      Loop *L;          // non-null: hint goes in L's preheader
      Demand D;
      unsigned Line;
    };
    json::Array Log;
    unsigned NumHints = 0, NumRegions = 0, NumRedundant = 0;
    bool CFGChanged = false; // preheaders inserted

    std::vector<Function *> Funcs;
    for (Function &F : M)
      if (!F.isDeclaration())
        Funcs.push_back(&F);
    for (Function *F : Funcs) {
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);
      DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
      std::vector<Planned> Plan;
      // Innermost loops.
      for (Loop *L : LI.getLoopsInPreorder()) {
        if (!L->isInnermost())
          continue;
        LoopBlocksRPO RPO(L);
        RPO.perform(&LI);
        std::vector<BasicBlock *> Blocks(RPO.begin(), RPO.end());
        std::vector<Node> Seq = buildSequence(Blocks, L, TM, JCopies);
        if (Seq.empty())
          continue;
        Demand D = analyze(Seq, TM, Full, SetIQ);
        unsigned Line = L->getStartLoc() ? L->getStartLoc().getLine() : 0;
        // The hint goes at the end of the preheader; create it now so that
        // the dataflow below sees the final CFG.
        BasicBlock *PH = L->getLoopPreheader();
        if (!PH) {
          PH = InsertPreheaderForLoop(L, &DT, &LI, nullptr, false);
          CFGChanged |= PH != nullptr;
        }
        if (!PH)
          continue;
        Plan.push_back({PH, L, D, Line});
      }
      // Blocks outside innermost loops.
      ReversePostOrderTraversal<Function *> FRPO(F);
      for (BasicBlock *BB : FRPO) {
        Loop *L = LI.getLoopFor(BB);
        if (L && L->isInnermost())
          continue;
        unsigned N = 0;
        for (Instruction &I : *BB)
          N += occupies(I);
        if (N < JMinBlock)
          continue;
        BasicBlock *One[] = {BB};
        std::vector<Node> Seq = buildSequence(One, nullptr, TM, JCopies);
        Demand D = analyze(Seq, TM, Full, SetIQ);
        unsigned Line = 0;
        for (Instruction &I : *BB)
          if (I.getDebugLoc()) {
            Line = I.getDebugLoc().getLine();
            break;
          }
        Plan.push_back({BB, nullptr, D, Line});
      }
      NumRegions += Plan.size();

      // Redundancy elimination: forward must-dataflow of the setting in force.
      // State: -1 unknown, else hint value. A block hint sets the state at the
      // block entry; a loop hint sets it at the end of the loop's preheader
      // (so a back edge carries whatever the body leaves in force). A call to
      // a function that may contain hints (defined in this module, or an
      // indirect call) leaves the state unknown.
      DenseMap<BasicBlock *, unsigned> EntryHint, ExitHint;
      for (Planned &P : Plan)
        (P.L ? ExitHint : EntryHint)[P.At] = P.D.Hint;
      DenseMap<BasicBlock *, bool> Kills;
      for (BasicBlock &BB : *F)
        for (Instruction &I : BB)
          if (auto *CB = dyn_cast<CallBase>(&I)) {
            if (CB->isInlineAsm() || isa<IntrinsicInst>(CB))
              continue;
            Function *Callee = CB->getCalledFunction();
            if (!Callee || !Callee->isDeclaration())
              Kills[&BB] = true;
          }
      DenseMap<BasicBlock *, int> In, Out;
      // State just before the terminator, ignoring the block's exit hint.
      auto BeforeExit = [&](BasicBlock *BB) -> int {
        if (Kills.lookup(BB))
          return -1;
        auto It = EntryHint.find(BB);
        return It != EntryHint.end() ? (int)It->second : In[BB];
      };
      bool Changed = true;
      const int Top = -2; // optimistic initial value
      for (BasicBlock &BB : *F)
        Out[&BB] = Top;
      while (Changed) {
        Changed = false;
        for (BasicBlock *BB : FRPO) {
          int V = Top;
          bool First = true;
          if (BB == &F->getEntryBlock())
            V = -1, First = false;
          for (BasicBlock *P : predecessors(BB)) {
            int PV = Out[P];
            if (PV == Top)
              continue;
            if (First) {
              V = PV;
              First = false;
            } else if (V != PV)
              V = -1;
          }
          In[BB] = V;
          auto XIt = ExitHint.find(BB);
          int O = XIt != ExitHint.end() ? (int)XIt->second : BeforeExit(BB);
          if (O != Out[BB]) {
            Out[BB] = O;
            Changed = true;
          }
        }
      }
      for (Planned &P : Plan) {
        // Loops: the state before the hint at the end of the preheader.
        int Incoming = P.L ? BeforeExit(P.At) : In[P.At];
        if (Incoming == Top)
          Incoming = -1;
        bool Redundant = Incoming == (int)P.D.Hint;
        json::Object E{{"function", F->getName()},
                       {"line", (int64_t)P.Line},
                       {"region", P.L ? "loop" : "block"},
                       {"insts", (int64_t)P.D.Insts},
                       {"iq_demand", (int64_t)P.D.IQ},
                       {"config", (int64_t)P.D.Config},
                       {"hint", (int64_t)P.D.Hint},
                       {"redundant", Redundant}};
        Log.push_back(std::move(E));
        if (Redundant) {
          ++NumRedundant;
          continue;
        }
        Instruction *IP =
            P.L ? P.At->getTerminator() : &*P.At->getFirstInsertionPt();
        if (Emitter.emit(IP, Kind, P.D.Hint))
          ++NumHints;
        if (JVerbose)
          errs() << "jones-iq: " << F->getName() << " line " << P.Line
                 << (P.L ? " loop" : " block") << " insts=" << P.D.Insts << " IQ=" << P.D.IQ
                 << " config=" << P.D.Config << " -> " << Emitter.describe(Kind, P.D.Hint)
                 << "\n";
      }
    }
    double Ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
    if (!JOutDir.empty()) {
      sys::fs::create_directories(JOutDir);
      SmallString<256> P(JOutDir.getValue());
      sys::path::append(P, kernelName(M) + (Full ? ".jones_full.json" : ".jones.json"));
      std::error_code EC;
      raw_fd_ostream OS(P, EC, sys::fs::OF_Text);
      if (!EC) {
        json::OStream J(OS, 2);
        J.object([&] {
          J.attribute("kernel", kernelName(M));
          J.attribute("baseline", "B6 Jones et al. HPCA'05");
          J.attribute("mode", Full ? "full" : "iq");
          J.attribute("iq_only", !Full);
          J.attribute("encoding", Full ? "setwin" : JEnc.getValue());
          J.attribute("emit", emitModeName(EM));
          J.attribute("target", TM.Name);
          J.attribute("regions", (int64_t)NumRegions);
          J.attribute("hints", (int64_t)NumHints);
          J.attribute("redundant_removed", (int64_t)NumRedundant);
          J.attribute("compile_time_ms", Ms);
          J.attributeArray("log", [&] {
            for (json::Value &V : Log)
              J.value(V);
          });
        });
        OS << "\n";
      }
    }
    return NumHints || CFGChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

/// @brief Plugin descriptor of JonesIQ.so (pipeline name `jones-iq` and the
///        optimizer-last hook).
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "JonesIQ", LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "jones-iq") {
                    MPM.addPass(JonesIQPass());
                    return true;
                  }
                  return false;
                });
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel, ThinOrFullLTOPhase Phase) {
                  if (!JAuto || Phase == ThinOrFullLTOPhase::ThinLTOPreLink ||
                      Phase == ThinOrFullLTOPhase::FullLTOPreLink)
                    return;
                  MPM.addPass(JonesIQPass());
                });
          }};
}
