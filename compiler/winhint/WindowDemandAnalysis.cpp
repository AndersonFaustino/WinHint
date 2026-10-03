//===- WindowDemandAnalysis.cpp - WinHint window-demand model -------------===//
/**
 * @file
 * @brief Implementation of WindowDemandAnalysis: per-instruction costs, loop
 *        tree and trip counts, dependence-DAG critical path and recurrences,
 *        memory-access grouping, footprints / reuse distances, and the W*
 *        window model; plus execCost(), the libm summary table and the printer.
 */
#include "WindowDemandAnalysis.h"
#include "Options.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <optional>

using namespace llvm;

namespace winhint {

AnalysisKey WindowDemandAnalysis::Key;

// Documented in WindowDemandAnalysis.h.
const char *accessClassName(AccessClass C) {
  switch (C) {
  case AccessClass::Streaming:
    return "stream";
  case AccessClass::Strided:
    return "strided";
  case AccessClass::Invariant:
    return "invariant";
  case AccessClass::Indirect:
    return "indirect";
  case AccessClass::Chase:
    return "chase";
  }
  return "?";
}

//===----------------------------------------------------------------------===//
// Cost model shared with HintPlacement.
//===----------------------------------------------------------------------===//

// Documented in WindowDemandAnalysis.h.
double execCost(double Insts, double WStar, unsigned C, const TargetModel &TM,
                double EnergyWeight) {
  if (Insts <= 0)
    return 0;
  const WindowConfig &Cfg = TM.Window[std::min<size_t>(C, TM.Window.size() - 1)];
  double IPC0 = std::max(1.0, TM.IssueWidth * 0.5);
  double Eff = WStar <= 0 ? 1.0 : std::min(1.0, (double)Cfg.ROB / WStar);
  double T = Insts / (IPC0 * Eff);
  return T * (1.0 + EnergyWeight * (double)Cfg.ROB / (double)TM.wMax());
}

//===----------------------------------------------------------------------===//
// Library-call summaries (libm), used inside the dependence DAG.
//===----------------------------------------------------------------------===//

// Documented in WindowDemandAnalysis.h. Table entries: {name, insts, latency}.
bool getLibCallInfo(StringRef Name, LibCallInfo &Info) {
  StringRef N = Name;
  bool Intr = N.consume_front("llvm.");
  if (Intr) {
    // llvm.exp.f32 -> expf, llvm.exp.f64 -> exp; vectors by element type
    // (llvm.exp.v4f32 -> expf, llvm.exp.v2f64 / nxv2f64 -> exp).
    auto P = N.split('.');
    std::string Base = P.first.str();
    StringRef Ty = P.second;
    Ty.consume_front("nx");
    if (Ty.consume_front("v"))
      Ty = Ty.drop_while([](char Ch) { return isDigit(Ch); });
    if (Ty.starts_with("f32"))
      Base += "f";
    static thread_local std::string Buf;
    Buf = Base;
    N = Buf;
  }
  static const struct {
    const char *N;
    double I, L;
  } Tab[] = {
      {"expf", 22, 30},   {"exp", 26, 36},    {"exp2f", 20, 28},  {"exp2", 24, 32},
      {"expm1f", 26, 34}, {"logf", 22, 30},   {"log", 26, 36},    {"log2f", 22, 30},
      {"log10f", 24, 32}, {"log1pf", 26, 34}, {"tanhf", 34, 44},  {"tanh", 40, 52},
      {"sinf", 30, 40},   {"cosf", 30, 40},   {"sin", 36, 48},    {"cos", 36, 48},
      {"powf", 55, 70},   {"pow", 65, 85},    {"erff", 36, 46},   {"erf", 42, 54},
      {"atanf", 30, 40},  {"atan2f", 40, 50}, {"floorf", 4, 4},   {"ceilf", 4, 4},
      {"roundf", 6, 6},   {"fabsf", 1, 1},    {"fabs", 1, 1},     {"fmaxf", 1, 2},
      {"fminf", 1, 2},    {"fmax", 1, 2},     {"fmin", 1, 2},     {"sqrtf", 1, -1},
      {"sqrt", 1, -1},    {"maxnumf", 1, 2},  {"minnumf", 1, 2},  {"maximumf", 1, 2},
      {"minimumf", 1, 2}, {"copysignf", 1, 1}, {"fmuladdf", 1, -2}, {"fmaf", 1, -2},
      {"fmuladd", 1, -2}, {"fma", 1, -2},
  };
  for (auto &E : Tab)
    if (N == E.N) {
      Info = {E.I, E.L};
      return true;
    }
  return false;
}

//===----------------------------------------------------------------------===//
// Per-instruction latency and dynamic-instruction weight.
//===----------------------------------------------------------------------===//

namespace {

/// Model cost of one IR instruction.
struct InstCost {
  double Lat = 0;   ///< latency on the dependence DAG (cycles)
  double Count = 0; ///< dynamic machine instructions it expands to
  double FpDivOcc = 0, IntDivOcc = 0; ///< occupancy of non-pipelined units
  const Function *DefinedCallee = nullptr; ///< call to a body in the module
};

/// @brief Latency, instruction count and divider occupancy of I.
///
/// PHIs, bitcasts, addrspacecasts, freeze, constant-index GEPs, unconditional branches and
/// debug/lifetime/assume intrinsics count 0; summarized libm calls use
/// getLibCallInfo(); other calls count 5 instructions at OpLatencies::Call.
/// @param I  Instruction.
/// @param TM Target model (latencies).
/// @return The cost; DefinedCallee is set for calls to functions defined in the module.
InstCost instCost(const Instruction &I, const TargetModel &TM) {
  const OpLatencies &L = TM.Lat;
  InstCost C;
  C.Count = 1;
  switch (I.getOpcode()) {
  case Instruction::PHI:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
  case Instruction::Freeze:
    C.Count = 0;
    return C;
  case Instruction::GetElementPtr: {
    auto *G = cast<GetElementPtrInst>(&I);
    if (G->hasAllConstantIndices()) {
      C.Count = 0; // folded into the addressing mode
      return C;
    }
    C.Lat = L.IntAlu;
    C.Count = 2; // shift + add
    return C;
  }
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::ICmp:
  case Instruction::Select:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::Trunc:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::ExtractElement:
  case Instruction::InsertElement:
  case Instruction::ShuffleVector:
  case Instruction::ExtractValue:
  case Instruction::InsertValue:
    C.Lat = L.IntAlu;
    return C;
  case Instruction::Mul:
    C.Lat = L.IntMul;
    return C;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
    C.Lat = L.IntDiv;
    C.IntDivOcc = L.IntDiv;
    return C;
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FNeg:
  case Instruction::FCmp:
    C.Lat = L.FpAdd;
    return C;
  case Instruction::FMul:
    C.Lat = L.FpMul;
    return C;
  case Instruction::FDiv:
  case Instruction::FRem:
    C.Lat = L.FpDiv;
    C.FpDivOcc = L.FpDiv;
    return C;
  case Instruction::FPToSI:
  case Instruction::FPToUI:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
    C.Lat = L.FpCvt;
    return C;
  case Instruction::Load:
    C.Lat = TM.l1Latency();
    return C;
  case Instruction::Store:
    C.Lat = L.Store;
    return C;
#if LLVM_VERSION_MAJOR >= 23
  case Instruction::UncondBr:
    C.Count = 0; // folded by layout / fused with the compare
    C.Lat = L.Branch;
    return C;
  case Instruction::CondBr:
    C.Lat = L.Branch;
    return C;
#else
  case Instruction::Br:
    C.Count = cast<BranchInst>(&I)->isConditional() ? 1 : 0;
    C.Lat = L.Branch;
    return C;
#endif
  case Instruction::Switch:
  case Instruction::IndirectBr:
  case Instruction::Ret:
  case Instruction::Unreachable:
    C.Lat = L.Branch;
    return C;
  case Instruction::Call:
  case Instruction::Invoke: {
    auto &CB = cast<CallBase>(I);
    if (isa<DbgInfoIntrinsic>(&I) || I.isLifetimeStartOrEnd() || isa<AssumeInst>(&I) ||
        isa<PseudoProbeInst>(&I)) {
      C.Count = 0;
      return C;
    }
    if (CB.isInlineAsm()) {
      C.Lat = 1;
      return C;
    }
    const Function *Callee = CB.getCalledFunction();
    LibCallInfo LI;
    if (Callee && getLibCallInfo(Callee->getName(), LI)) {
      C.Count = LI.Insts;
      C.Lat = LI.Latency == -1 ? L.FpSqrt : LI.Latency == -2 ? L.FpFma : LI.Latency;
      if (LI.Latency == -1)
        C.FpDivOcc = L.FpSqrt;
      return C;
    }
    if (Callee && Callee->isIntrinsic()) {
      C.Lat = L.IntAlu;
      return C;
    }
    C.Count = 5; // call/ret + argument moves; callee body is summarized separately
    C.Lat = L.Call;
    if (Callee && !Callee->isDeclaration())
      C.DefinedCallee = Callee;
    return C;
  }
  default:
    C.Lat = L.IntAlu;
    return C;
  }
}

//===----------------------------------------------------------------------===//
// Call-site argument bounds (context for symbolic trip counts and strides).
//===----------------------------------------------------------------------===//

/// @brief Collect the constant values V may take (through select, phi,
///        zext/sext/trunc and add/sub/mul/shl of small constant sets; a shl
///        by an amount outside [0, 62) yields its left operand unchanged).
/// @param[in]     V     Value to analyze.
/// @param[out]    Out   Possible values (sign-extended), appended.
/// @param[in]     Depth Remaining recursion depth.
/// @param[in,out] Seen  Visited values (cycle guard).
/// @return false if some reachable leaf is not a constant or the depth or
///         product-size limit (16) is exceeded.
bool collectConsts(const Value *V, SmallVectorImpl<int64_t> &Out, unsigned Depth,
                   SmallPtrSetImpl<const Value *> &Seen) {
  if (!Seen.insert(V).second)
    return true;
  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    Out.push_back(CI->getSExtValue());
    return true;
  }
  if (Depth == 0)
    return false;
  if (auto *S = dyn_cast<SelectInst>(V))
    return collectConsts(S->getTrueValue(), Out, Depth - 1, Seen) &&
           collectConsts(S->getFalseValue(), Out, Depth - 1, Seen);
  if (auto *P = dyn_cast<PHINode>(V)) {
    for (const Value *In : P->incoming_values())
      if (!collectConsts(In, Out, Depth - 1, Seen))
        return false;
    return true;
  }
  if (auto *C = dyn_cast<CastInst>(V))
    if (C->getOpcode() == Instruction::ZExt || C->getOpcode() == Instruction::SExt ||
        C->getOpcode() == Instruction::Trunc)
      return collectConsts(C->getOperand(0), Out, Depth - 1, Seen);
  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    // c1 op c2 over small constant sets (e.g. S * HIDDEN with S = select).
    SmallVector<int64_t, 4> A, B;
    SmallPtrSet<const Value *, 8> S1, S2;
    if (!collectConsts(BO->getOperand(0), A, Depth - 1, S1) ||
        !collectConsts(BO->getOperand(1), B, Depth - 1, S2) || A.size() * B.size() > 16)
      return false;
    for (int64_t X : A)
      for (int64_t Y : B) {
        switch (BO->getOpcode()) {
        case Instruction::Add:
          Out.push_back(X + Y);
          break;
        case Instruction::Sub:
          Out.push_back(X - Y);
          break;
        case Instruction::Mul:
          Out.push_back(X * Y);
          break;
        case Instruction::Shl:
          // Only in-range shift amounts (a negative or too large Y is UB in C++).
          Out.push_back(Y >= 0 && Y < 62 ? (int64_t)((uint64_t)X << Y) : X);
          break;
        default:
          return false;
        }
      }
    return true;
  }
  return false;
}

/// @brief For each integer argument, the maximum constant value it takes over
///        all direct call sites in the module (if every call site is analyzable).
/// @param F  Function whose arguments are bounded.
/// @param SE ScalarEvolution of F (creates the constant SCEVs).
/// @return Argument -> constant SCEV; empty if F has no callers or any use is
///         not a direct call.
ValueToSCEVMapTy argumentBounds(Function &F, ScalarEvolution &SE) {
  ValueToSCEVMapTy Map;
  SmallVector<CallBase *, 8> Sites;
  for (User *U : F.users()) {
    auto *CB = dyn_cast<CallBase>(U);
    if (!CB || CB->getCalledFunction() != &F)
      return Map; // address taken or indirect use: no context
    Sites.push_back(CB);
  }
  if (Sites.empty())
    return Map;
  for (Argument &A : F.args()) {
    if (!A.getType()->isIntegerTy() || A.getType()->getIntegerBitWidth() > 64)
      continue;
    bool OK = true;
    int64_t Max = INT64_MIN;
    for (CallBase *CB : Sites) {
      SmallVector<int64_t, 4> Vals;
      SmallPtrSet<const Value *, 8> Seen;
      if (!collectConsts(CB->getArgOperand(A.getArgNo()), Vals, 6, Seen) || Vals.empty()) {
        OK = false;
        break;
      }
      for (int64_t V : Vals)
        Max = std::max(Max, V);
    }
    if (OK)
      Map[&A] = SE.getConstant(A.getType(), (uint64_t)Max, /*isSigned=*/true);
  }
  return Map;
}

/// Evaluates SCEVs to constants, optionally substituting argument bounds.
struct Evaluator {
  ScalarEvolution &SE;      ///< scalar evolution of the function
  ValueToSCEVMapTy &Map;    ///< argument -> constant bound (argumentBounds())
  bool UsedContext = false; ///< set when a value needed the argument bounds

  /// @brief Constant value of S, directly or after rewriting the arguments
  ///        with Map; std::nullopt otherwise.
  std::optional<double> eval(const SCEV *S) {
    if (!S || isa<SCEVCouldNotCompute>(S))
      return std::nullopt;
    if (auto *C = dyn_cast<SCEVConstant>(S))
      return (double)C->getAPInt().getSExtValue();
    if (!Map.empty()) {
      const SCEV *R = SCEVParameterRewriter::rewrite(S, SE, Map);
      if (auto *C = dyn_cast<SCEVConstant>(R)) {
        UsedContext = true;
        return (double)C->getAPInt().getSExtValue();
      }
    }
    return std::nullopt;
  }
};

//===----------------------------------------------------------------------===//
// Address slices: indirect and pointer-chasing loads.
//===----------------------------------------------------------------------===//

/// @brief Collect the in-loop backward slice of V. Stops at header PHIs of L
///        (recorded in Phis) and at values defined outside L.
/// @param[in]     V       Root value.
/// @param[in]     L       Loop bounding the slice.
/// @param[in,out] Seen    Instructions visited.
/// @param[in,out] Phis    Header PHIs of L reached.
/// @param[in,out] HasLoad Set if the slice contains a load other than V.
/// @param[in]     Budget  Maximum number of instructions visited.
void sliceInLoop(const Value *V, const Loop *L, SmallPtrSetImpl<const Instruction *> &Seen,
                 SmallPtrSetImpl<const PHINode *> &Phis, bool &HasLoad, unsigned Budget = 256) {
  SmallVector<const Value *, 16> WL{V};
  while (!WL.empty() && Budget) {
    const Value *X = WL.pop_back_val();
    auto *I = dyn_cast<Instruction>(X);
    if (!I || !L->contains(I) || !Seen.insert(I).second)
      continue;
    --Budget;
    if (auto *P = dyn_cast<PHINode>(I))
      if (P->getParent() == L->getHeader()) {
        Phis.insert(P);
        continue;
      }
    if (isa<LoadInst>(I) && I != V)
      HasLoad = true;
    for (const Value *Op : I->operands())
      WL.push_back(Op);
  }
}

/// @brief First cache level whose effective capacity (size *
///        CacheEffectiveFraction) holds Bytes; Caches.size() (memory) if none.
unsigned levelFor(double Bytes, const TargetModel &TM) {
  for (unsigned I = 0; I < TM.Caches.size(); ++I)
    if (Bytes <= TM.Caches[I].SizeBytes * TM.CacheEffectiveFraction)
      return I;
  return TM.Caches.size();
}

/// @brief Format a byte count as "123B", "1.5KiB", "2.0MiB" or "unbounded".
std::string fmtBytes(double B) {
  std::string S;
  raw_string_ostream OS(S);
  if (std::isinf(B))
    OS << "unbounded";
  else if (B >= 1024.0 * 1024)
    OS << format("%.1fMiB", B / (1024.0 * 1024));
  else if (B >= 1024)
    OS << format("%.1fKiB", B / 1024);
  else
    OS << format("%.0fB", B);
  return S;
}

/// One load or store before grouping.
struct RawAccess {
  Instruction *I; ///< the load or store
  bool IsLoad;    ///< load (true) or store
  double Elem;    ///< access size in bytes (at least 1)
  const SCEV *Base = nullptr;    ///< start after peeling AddRecs of ancestors
  const SCEV *PtrBase = nullptr; ///< SE.getPointerBase
  std::vector<const SCEV *> Steps; ///< per ancestor (index 0 = own loop), null = 0
  AccessClass Class = AccessClass::Streaming; ///< initial class (Strided is assigned later)
  double ObjectBytes = -1; ///< known underlying object size
};

} // namespace

//===----------------------------------------------------------------------===//
// The analysis.
//===----------------------------------------------------------------------===//

// Documented in WindowDemandAnalysis.h.
bool FunctionDemand::invalidate(Function &, const PreservedAnalyses &PA,
                                FunctionAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<WindowDemandAnalysis>();
  return !PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Function>>();
}

// Documented in WindowDemandAnalysis.h.
FunctionDemand WindowDemandAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  const TargetModel &TM = getTargetModel();
  FunctionDemand R;
  R.F = &F;
  if (F.isDeclaration())
    return R;

  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  DependenceInfo &DI = FAM.getResult<DependenceAnalysis>(F);
  const DataLayout &DL = F.getParent()->getDataLayout();
  const double Line = TM.LineSize;

  ValueToSCEVMapTy ArgMap = argumentBounds(F, SE);

  // RPO numbering (program order for siblings and calls).
  DenseMap<const BasicBlock *, unsigned> RPO;
  {
    ReversePostOrderTraversal<Function *> RPOT(&F);
    unsigned N = 0;
    for (BasicBlock *BB : RPOT)
      RPO[BB] = N++;
  }

  // ---- Build the loop tree in preorder ------------------------------------
  SmallVector<Loop *, 16> Pre = LI.getLoopsInPreorder();
  for (Loop *L : Pre) {
    auto D = std::make_unique<LoopDemand>();
    D->L = L;
    D->Depth = L->getLoopDepth();
    D->IndexInFunction = R.Loops.size();
    D->RPOIndex = RPO.lookup(L->getHeader());
    if (DebugLoc DLc = L->getStartLoc())
      D->Line = DLc.getLine();
    D->HeaderName = L->getHeader()->hasName() ? L->getHeader()->getName().str()
                                              : ("loop" + std::to_string(D->IndexInFunction));
    R.ByLoop[L] = D.get();
    R.Loops.push_back(std::move(D));
  }
  for (auto &DP : R.Loops) {
    LoopDemand *D = DP.get();
    if (Loop *P = D->L->getParentLoop())
      D->Parent = R.ByLoop.lookup(P);
    if (D->Parent)
      D->Parent->Children.push_back(D);
    else
      R.TopLevel.push_back(D);
  }
  auto ByRPO = [](LoopDemand *A, LoopDemand *B) { return A->RPOIndex < B->RPOIndex; };
  std::stable_sort(R.TopLevel.begin(), R.TopLevel.end(), ByRPO);
  for (auto &DP : R.Loops)
    std::stable_sort(DP->Children.begin(), DP->Children.end(), ByRPO);

  // ---- Trip counts ---------------------------------------------------------
  for (auto &DP : R.Loops) {
    LoopDemand &D = *DP;
    Evaluator Ev{SE, ArgMap};
    std::optional<double> BTC = Ev.eval(SE.getBackedgeTakenCount(D.L));
    bool FromMax = false;
    if (!BTC) {
      // Only an upper bound is known. A bound that merely reflects the range
      // of the induction variable's type (e.g. 2^31 - 9 for `i < n - 8` with
      // an int n) says nothing about the trip count: treat it as unknown.
      FromMax = true;
      BTC = Ev.eval(SE.getSymbolicMaxBackedgeTakenCount(D.L));
      if (!BTC)
        BTC = Ev.eval(SE.getConstantMaxBackedgeTakenCount(D.L));
      if (BTC && *BTC >= (double)(1u << 30))
        BTC.reset();
    }
    if (BTC && *BTC >= 0 && *BTC < 1e10) {
      D.Trip = *BTC + 1;
      D.TripKnown = !Ev.UsedContext && !FromMax;
      D.TripBounded = !D.TripKnown;
    } else {
      D.Trip = OptUnknownTrip;
      D.Conservative = true;
      D.Notes += "unknown-trip ";
    }
  }

  // ---- Own instructions, calls, CP, RecMII ---------------------------------
  // Straight-line code of the function.
  for (BasicBlock &BB : F) {
    if (LI.getLoopFor(&BB))
      continue;
    for (Instruction &I : BB) {
      InstCost C = instCost(I, TM);
      R.StraightInsts += C.Count;
      if (C.DefinedCallee)
        R.StraightCalls.push_back({cast<CallBase>(&I), RPO.lookup(&BB)});
    }
  }

  for (auto &DP : R.Loops) {
    LoopDemand &D = *DP;
    Loop *L = D.L;
    LoopBlocksRPO LRPO(L);
    LRPO.perform(&LI);
    DenseMap<const Instruction *, double> Finish;
    std::vector<Instruction *> Order;
    for (BasicBlock *BB : LRPO) {
      if (LI.getLoopFor(BB) != L)
        continue;
      for (Instruction &I : *BB)
        Order.push_back(&I);
    }
    double CP = 0, FpDivOcc = 0, IntDivOcc = 0;
    for (Instruction *I : Order) {
      InstCost C = instCost(*I, TM);
      D.OwnInsts += C.Count;
      FpDivOcc += C.FpDivOcc;
      IntDivOcc += C.IntDivOcc;
      if (C.DefinedCallee)
        D.Calls.push_back({cast<CallBase>(I), RPO.lookup(I->getParent())});
      if (isa<LoadInst>(I))
        ++D.NumLoads;
      if (isa<StoreInst>(I))
        ++D.NumStores;
      double Ready = 0;
      if (!isa<PHINode>(I))
        for (Value *Op : I->operands())
          if (auto *OI = dyn_cast<Instruction>(Op)) {
            auto It = Finish.find(OI);
            if (It != Finish.end())
              Ready = std::max(Ready, It->second);
          }
      double Fin = isa<PHINode>(I) ? 0 : Ready + C.Lat;
      Finish[I] = Fin;
      CP = std::max(CP, Fin);
    }
    D.CP = CP;

    // Recurrences through header PHIs (register-carried).
    double Rec = 0;
    unsigned Budget = 64;
    for (PHINode &P : L->getHeader()->phis()) {
      if (!Budget--)
        break;
      BasicBlock *Latch = L->getLoopLatch();
      if (!Latch || P.getBasicBlockIndex(Latch) < 0)
        continue;
      Value *Back = P.getIncomingValueForBlock(Latch);
      DenseMap<const Instruction *, double> Dist;
      Dist[&P] = 0;
      for (Instruction *I : Order) {
        if (isa<PHINode>(I))
          continue;
        double Best = -1;
        for (Value *Op : I->operands())
          if (auto *OI = dyn_cast<Instruction>(Op)) {
            auto It = Dist.find(OI);
            if (It != Dist.end())
              Best = std::max(Best, It->second);
          }
        if (Best >= 0)
          Dist[I] = Best + instCost(*I, TM).Lat;
      }
      if (auto *BI = dyn_cast<Instruction>(Back)) {
        auto It = Dist.find(BI);
        if (It != Dist.end())
          Rec = std::max(Rec, It->second);
      }
    }
    D.RecMII = Rec;
    D.OwnInsts = std::max(D.OwnInsts, 1.0);
    // Initiation interval: recurrences, issue width, non-pipelined units.
    D.II = std::max({D.RecMII, D.OwnInsts / TM.IssueWidth, FpDivOcc / TM.FpDivUnits,
                     IntDivOcc / TM.IntDivUnits, 1.0});
  }

  // ---- Dynamic instruction counts (bottom-up) ------------------------------
  for (auto It = R.Loops.rbegin(); It != R.Loops.rend(); ++It) {
    LoopDemand &D = **It;
    double Inner = 0;
    for (LoopDemand *C : D.Children)
      Inner += C->DynInsts;
    D.DynInsts = D.Trip * (D.OwnInsts + Inner);
  }

  // ---- Memory accesses ------------------------------------------------------
  // Ancestor chain helper: index 0 = own loop.
  auto Ancestors = [&](LoopDemand *D) {
    std::vector<LoopDemand *> A;
    for (LoopDemand *X = D; X; X = X->Parent)
      A.push_back(X);
    return A;
  };

  DenseMap<LoopDemand *, std::vector<RawAccess>> Raw;
  for (auto &DP : R.Loops) {
    LoopDemand &D = *DP;
    Loop *L = D.L;
    std::vector<LoopDemand *> Anc = Ancestors(&D);
    for (BasicBlock *BB : L->blocks()) {
      if (LI.getLoopFor(BB) != L)
        continue;
      for (Instruction &I : *BB) {
        if (!isa<LoadInst>(I) && !isa<StoreInst>(I))
          continue;
        Value *Ptr = getLoadStorePointerOperand(&I);
        Type *Ty = getLoadStoreType(&I);
        RawAccess A;
        A.I = &I;
        A.IsLoad = isa<LoadInst>(I);
        A.Elem = std::max<double>(1, DL.getTypeStoreSize(Ty).getKnownMinValue());
        A.Steps.assign(Anc.size(), nullptr);
        const SCEV *S = SE.getSCEV(Ptr);
        // Peel AddRecs of the enclosing loops (innermost first).
        while (auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
          const Loop *AL = AR->getLoop();
          auto Pos = std::find_if(Anc.begin(), Anc.end(),
                                  [&](LoopDemand *X) { return X->L == AL; });
          if (Pos == Anc.end() || !AR->isAffine())
            break;
          A.Steps[Pos - Anc.begin()] = AR->getStepRecurrence(SE);
          S = AR->getStart();
        }
        A.Base = S;
        A.PtrBase = SE.getPointerBase(S);
        const Value *Obj = getUnderlyingObject(Ptr);
        if (auto *GV = dyn_cast<GlobalVariable>(Obj))
          if (GV->getValueType()->isSized())
            A.ObjectBytes = DL.getTypeAllocSize(GV->getValueType()).getFixedValue();
        if (auto *AI = dyn_cast<AllocaInst>(Obj))
          if (auto Sz = AI->getAllocationSize(DL))
            A.ObjectBytes = Sz->getFixedValue();

        if (!SE.isLoopInvariant(S, L)) {
          // Not affine in the own loop: indirect or pointer chase.
          SmallPtrSet<const Instruction *, 32> Seen;
          SmallPtrSet<const PHINode *, 4> Phis;
          bool HasLoad = false;
          sliceInLoop(Ptr, L, Seen, Phis, HasLoad);
          bool Chase = false;
          if (BasicBlock *Latch = L->getLoopLatch())
            for (const PHINode *P : Phis) {
              if (P->getBasicBlockIndex(Latch) < 0)
                continue;
              SmallPtrSet<const Instruction *, 32> Seen2;
              SmallPtrSet<const PHINode *, 4> Phis2;
              bool BackHasLoad = false;
              const Value *Back = P->getIncomingValueForBlock(Latch);
              if (isa<LoadInst>(Back) && L->contains(cast<Instruction>(Back)))
                BackHasLoad = true;
              sliceInLoop(Back, L, Seen2, Phis2, BackHasLoad);
              if (BackHasLoad && Phis2.count(P))
                Chase = true;
            }
          A.Class = Chase ? AccessClass::Chase : AccessClass::Indirect;
        } else if (!A.Steps[0]) {
          A.Class = AccessClass::Invariant;
        }
        Raw[&D].push_back(A);
      }
    }
  }

  // Group accesses and compute strides/footprints.
  for (auto &DP : R.Loops) {
    LoopDemand &D = *DP;
    std::vector<LoopDemand *> Anc = Ancestors(&D);
    std::vector<RawAccess> &Acc = Raw[&D];
    std::vector<bool> Used(Acc.size(), false);
    for (size_t I = 0; I < Acc.size(); ++I) {
      if (Used[I])
        continue;
      Used[I] = true;
      RawAccess &Lead = Acc[I];
      AccessGroup G;
      G.Leader = Lead.I;
      G.IsLoad = Lead.IsLoad;
      G.Class = Lead.Class;
      G.ElemBytes = Lead.Elem;
      G.ObjectBytes = Lead.ObjectBytes;
      double MinOff = 0, MaxOff = Lead.Elem;
      if (Lead.Class != AccessClass::Indirect && Lead.Class != AccessClass::Chase)
        for (size_t J = I + 1; J < Acc.size(); ++J) {
          RawAccess &O = Acc[J];
          if (Used[J] || O.IsLoad != Lead.IsLoad || O.Class != Lead.Class ||
              O.PtrBase != Lead.PtrBase || O.Steps != Lead.Steps)
            continue;
          const SCEV *Diff = SE.getMinusSCEV(O.Base, Lead.Base);
          auto *C = dyn_cast<SCEVConstant>(Diff);
          if (!C)
            continue;
          double Off = (double)C->getAPInt().getSExtValue();
          if (std::fabs(Off) > 4 * Line)
            continue;
          Used[J] = true;
          ++G.Members;
          MinOff = std::min(MinOff, Off);
          MaxOff = std::max(MaxOff, Off + O.Elem);
        }
      G.SpanBytes = MaxOff - MinOff;
      // Strides per ancestor.
      Evaluator Ev{SE, ArgMap};
      G.Strides.assign(Anc.size(), 0);
      G.StrideIsZero.assign(Anc.size(), true);
      for (size_t K = 0; K < Anc.size(); ++K) {
        const SCEV *St = Lead.Steps[K];
        if (!St)
          continue;
        G.StrideIsZero[K] = false;
        if (auto V = Ev.eval(St))
          G.Strides[K] = std::fabs(*V);
        else {
          G.Strides[K] = 64 * Line; // unknown symbolic stride: assume large
          if (K == 0)
            G.StrideKnown = false;
        }
        if (G.Strides[K] == 0)
          G.StrideIsZero[K] = true;
      }
      if (G.Class == AccessClass::Indirect || G.Class == AccessClass::Chase) {
        // Treat as a fresh, line-granular access per iteration.
        G.StrideIsZero[0] = false;
        G.Strides[0] = Line;
      }
      G.InnerStride = G.Strides[0];
      if (G.Class == AccessClass::Streaming && !G.StrideIsZero[0] && G.InnerStride >= Line)
        G.Class = AccessClass::Strided;
      // Footprint over one entry of each ancestor.
      G.FootprintAt.assign(Anc.size(), 0);
      double Ne = 1, Span = G.SpanBytes, MinS = 0;
      for (size_t K = 0; K < Anc.size(); ++K) {
        double T = Anc[K]->Trip;
        // Unknown extent along a varying dimension: conservatively unbounded
        // (assume it streams from memory) unless the object size is known.
        if (!Anc[K]->TripKnown && !Anc[K]->TripBounded && !G.StrideIsZero[K])
          T = INFINITY;
        if (!G.StrideIsZero[K]) {
          Ne *= T;
          Span += G.Strides[K] * (T - 1);
          MinS = MinS == 0 ? G.Strides[K] : std::min(MinS, G.Strides[K]);
        }
        double Dense = Ne * std::max(G.SpanBytes, std::min(MinS == 0 ? G.SpanBytes : MinS, Line));
        double FP = std::min(Span, Dense);
        if (G.Class == AccessClass::Indirect || G.Class == AccessClass::Chase)
          FP = Dense; // unknown layout: every access may touch a new line
        if (Lead.ObjectBytes > 0)
          FP = std::min(FP, Lead.ObjectBytes);
        G.FootprintAt[K] = FP;
      }
      G.FootprintBytes = G.FootprintAt.back();
      D.Groups.push_back(std::move(G));
      if (Lead.Class == AccessClass::Indirect)
        ++D.NumIndirect;
      if (Lead.Class == AccessClass::Chase)
        ++D.NumChase;
    }
    for (AccessGroup &G : D.Groups)
      D.UnrollFactor = std::max(D.UnrollFactor, G.Members);
  }

  // Footprint of one iteration of each loop (sum over all groups beneath it).
  DenseMap<LoopDemand *, double> IterFP;
  for (auto &DP : R.Loops) {
    LoopDemand *D = DP.get();
    std::vector<LoopDemand *> Anc = Ancestors(D);
    for (AccessGroup &G : D->Groups)
      for (size_t K = 0; K < Anc.size(); ++K)
        IterFP[Anc[K]] += K == 0 ? G.SpanBytes : G.FootprintAt[K - 1];
  }
  for (LoopDemand *Top : R.TopLevel) {
    double Total = 0;
    SmallVector<LoopDemand *, 8> WL{Top};
    while (!WL.empty()) {
      LoopDemand *X = WL.pop_back_val();
      for (AccessGroup &G : X->Groups)
        Total += G.FootprintBytes;
      for (LoopDemand *C : X->Children)
        WL.push_back(C);
    }
    WL.push_back(Top);
    while (!WL.empty()) {
      LoopDemand *X = WL.pop_back_val();
      X->NestFootprint = Total;
      for (LoopDemand *C : X->Children)
        WL.push_back(C);
    }
  }

  // Reuse distance -> service level -> misses per iteration.
  for (auto &DP : R.Loops) {
    LoopDemand *D = DP.get();
    std::vector<LoopDemand *> Anc = Ancestors(D);
    for (AccessGroup &G : D->Groups) {
      double RD = -1;
      for (size_t K = 0; K < Anc.size(); ++K)
        if (G.StrideIsZero[K]) {
          RD = IterFP.lookup(Anc[K]);
          break;
        }
      G.ReuseDistanceBytes = RD;
      // No reuse inside the nest: the data was last touched elsewhere, so it
      // is found at the level that holds the whole object (if known).
      double Reach = RD >= 0 ? RD : std::max(D->NestFootprint, G.ObjectBytes);
      G.Level = levelFor(Reach, TM);
      if (G.Class == AccessClass::Indirect || G.Class == AccessClass::Chase) {
        // Unknown layout: bounded only by a known object size.
        double Obj = G.FootprintBytes;
        G.Level = levelFor(std::max(Obj, Reach), TM);
      }
      if (TM.StridePrefetcher && G.Class == AccessClass::Streaming && G.Level > 1)
        G.Level = 1;
      if (G.StrideIsZero[0] || G.Level == 0) {
        G.MissesPerIter = 0;
      } else if (G.Class == AccessClass::Indirect || G.Class == AccessClass::Chase) {
        G.MissesPerIter = G.Members;
      } else if (G.InnerStride < Line) {
        G.MissesPerIter = G.InnerStride / Line;
      } else {
        G.MissesPerIter = std::min<double>(G.Members, std::max(1.0, G.SpanBytes / Line));
      }
      if (!G.StrideKnown || D->Conservative)
        D->Conservative = true;
    }

    // Memory-carried recurrences (DependenceAnalysis): a load fed by a store
    // of an earlier iteration with a known distance is served by forwarding /
    // L1 and sits on the recurrence, so it does not add independent MLP.
    if (D->Children.empty()) {
      SmallVector<Instruction *, 16> Loads, Stores;
      for (BasicBlock *BB : D->L->blocks())
        for (Instruction &I : *BB) {
          if (isa<LoadInst>(I))
            Loads.push_back(&I);
          else if (isa<StoreInst>(I))
            Stores.push_back(&I);
        }
      if (!Stores.empty() && Loads.size() * Stores.size() <= 512) {
        for (AccessGroup &G : D->Groups) {
          if (!G.IsLoad)
            continue;
          for (Instruction *St : Stores) {
            auto Dep = DI.depends(St, const_cast<Instruction *>(G.Leader));
            if (!Dep || Dep->isConfused() || !Dep->isFlow() || Dep->isLoopIndependent())
              continue;
            unsigned Lv = Dep->getLevels();
            if (!Lv)
              continue;
            const SCEV *Dist = Dep->getDistance(Lv);
            auto *DC = dyn_cast_or_null<SCEVConstant>(Dist);
            if (DC && !DC->getAPInt().isZero()) {
              G.LoopCarried = true;
              ++D->NumLCD;
              break;
            }
          }
        }
      }
    }
  }

  // ---- The window model --------------------------------------------------
  const double IW = TM.IssueWidth;
  const double WMax = TM.wMax();
  const bool CPWidth = StringRef(OptCPModel) == "width";
  if (!CPWidth && StringRef(OptCPModel) != "ii")
    reportFatalUsageError("winhint: unknown -winhint-cp-model=" + StringRef(OptCPModel) +
                          " (expected ii or width)");
  for (auto &DP : R.Loops) {
    LoopDemand &D = *DP;
    double B = D.OwnInsts;
    double M = 0, LatSum = 0, ChaseLat = 0;
    bool HasChase = false;
    for (AccessGroup &G : D.Groups) {
      if (!G.IsLoad || G.Level == 0 || G.MissesPerIter <= 0)
        continue;
      double Lat = TM.levelLatency(G.Level);
      if (G.Class == AccessClass::Chase || G.LoopCarried) {
        HasChase |= G.Class == AccessClass::Chase;
        ChaseLat = std::max(ChaseLat, Lat);
        continue;
      }
      ++D.NumLongLat;
      M += G.MissesPerIter;
      LatSum += G.MissesPerIter * Lat;
    }
    D.MissesPerIter = M;
    // Issue rate sustained by the loop: bounded by the width and recurrences.
    double Rate = std::min(IW, B / D.II);
    // Dynamic instructions of one iteration, including the inner loops: the
    // distance between misses of the own blocks of an outer loop.
    double BIter = B;
    for (LoopDemand *C : D.Children)
      BIter += C->DynInsts;
    if (M > 0) {
      D.Lmem = LatSum / M;
      D.DIndep = BIter / M;
      // MLP_target = misses issued while one miss is outstanding (Little's
      // law: L_mem * rate instructions are issued during its latency, one
      // independent miss every D_indep of them), bounded by the MSHRs. It is
      // kept fractional: rounding it up to an integer would make W_mlp =
      // ceil(MLP) * D_indep, i.e. force a whole D_indep into the window even
      // when misses are rare (D_indep >> L_mem * rate) and only L_mem * rate
      // entries are needed to hide the one miss in flight.
      double MLP = TM.MLPTargetOverride > 0 ? TM.MLPTargetOverride
                                            : D.Lmem * Rate / D.DIndep;
      D.MLPTarget = std::min(MLP, (double)TM.L1DMSHRs);
      D.Wmlp = (unsigned)std::ceil(D.MLPTarget * D.DIndep - 1e-9);
      D.MemoryBound = D.Lmem * Rate > B;
    } else {
      D.Lmem = HasChase ? ChaseLat : TM.l1Latency();
      D.DIndep = INFINITY;
      D.MLPTarget = HasChase ? 1 : 0;
      D.Wmlp = HasChase ? (unsigned)std::ceil(B) : 0;
      D.MemoryBound = HasChase;
    }
    // W_cp = CP * issue rate (Little's law: an instruction stays in the window
    // for about the critical path, and the loop issues `Rate` per cycle).
    // Rate = issue_width unless recurrences / dividers bound the loop, so
    // this is the CP * issue_width of PROPOSAL §3.1 for issue-bound loops.
    // (-winhint-cp-model=width uses CP * issue_width unconditionally.)
    D.Wcp = (unsigned)std::ceil(D.CP * (CPWidth ? IW : Rate) - 1e-9);
    double W = std::max<double>(D.Wmlp, D.Wcp);
    D.WStar = (unsigned)std::min(WMax, std::max(1.0, W));
    D.Config = TM.configForW(D.WStar);
  }

  // Nest aggregates.
  double EW = OptEnergyWeight;
  for (LoopDemand *Top : R.TopLevel) {
    std::vector<std::pair<LoopDemand *, double>> Own; // (loop, dyn own insts per entry)
    SmallVector<std::pair<LoopDemand *, double>, 8> WL{{Top, 1.0}};
    while (!WL.empty()) {
      auto [X, Mult] = WL.pop_back_val();
      double Iters = Mult * X->Trip;
      Own.push_back({X, Iters * X->OwnInsts});
      for (LoopDemand *C : X->Children)
        WL.push_back({C, Iters});
    }
    double Num = 0, Den = 0;
    for (auto &[X, N] : Own) {
      Num += N * X->WStar;
      Den += N;
    }
    Top->NestWStar = Den > 0 ? Num / Den : Top->WStar;
    double Best = INFINITY;
    for (unsigned C = 0; C < TM.Window.size(); ++C) {
      double Cost = 0;
      for (auto &[X, N] : Own)
        Cost += execCost(N, X->WStar, C, TM, EW);
      if (Cost < Best - 1e-9) {
        Best = Cost;
        Top->NestConfig = C;
      }
    }
  }
  return R;
}

//===----------------------------------------------------------------------===//
// Printing.
//===----------------------------------------------------------------------===//

/// @brief Format V with one decimal, or "inf".
static std::string fmtNum(double V) {
  if (std::isinf(V))
    return "inf";
  std::string S;
  raw_string_ostream OS(S);
  OS << format("%.1f", V);
  return S;
}

// Documented in WindowDemandAnalysis.h.
void FunctionDemand::print(raw_ostream &OS, const TargetModel &TM) const {
  OS << "WinHint window demand for function '" << F->getName() << "' (" << Loops.size()
     << " loops, straight-line insts " << fmtNum(StraightInsts) << ")\n";
  for (const auto &DP : Loops) {
    const LoopDemand &D = *DP;
    OS.indent(2 * D.Depth) << "loop " << D.HeaderName << " line " << D.Line << " depth "
                           << D.Depth << (D.Children.empty() ? " (innermost)" : "") << "\n";
    OS.indent(2 * D.Depth + 2)
        << "trip=" << fmtNum(D.Trip)
        << (D.TripKnown ? " (exact)" : D.TripBounded ? " (bound)" : " (assumed)")
        << " body=" << fmtNum(D.OwnInsts) << " dyn_insts=" << fmtNum(D.DynInsts)
        << " footprint=" << fmtBytes(D.NestFootprint) << " unroll=" << D.UnrollFactor
        << " loads=" << D.NumLoads << " stores=" << D.NumStores << "\n";
    for (const AccessGroup &G : D.Groups) {
      OS.indent(2 * D.Depth + 4) << (G.IsLoad ? "load  " : "store ") << accessClassName(G.Class)
                                 << " x" << G.Members << " stride=" << fmtNum(G.InnerStride)
                                 << (G.StrideKnown ? "" : "?")
                                 << " fp=" << fmtBytes(G.FootprintBytes) << " reuse="
                                 << (G.ReuseDistanceBytes < 0 ? std::string("none")
                                                              : fmtBytes(G.ReuseDistanceBytes))
                                 << " level=" << TM.levelName(G.Level)
                                 << " misses/iter=" << format("%.3f", G.MissesPerIter)
                                 << (G.LoopCarried ? " loop-carried" : "") << "\n";
    }
    OS.indent(2 * D.Depth + 2)
        << "WINHINT fn=" << F->getName() << " loop=" << D.HeaderName << " line=" << D.Line
        << " depth=" << D.Depth << " L_mem=" << fmtNum(D.Lmem) << " D_indep=" << fmtNum(D.DIndep)
        << " MLP=" << fmtNum(D.MLPTarget) << " CP=" << fmtNum(D.CP)
        << " RecMII=" << fmtNum(D.RecMII) << " II=" << fmtNum(D.II) << " W_mlp=" << D.Wmlp
        << " W_cp=" << D.Wcp << " W*=" << D.WStar << " config=" << D.Config << " (ROB "
        << TM.Window[D.Config].ROB << ")" << (D.Conservative ? " conservative" : "") << "\n";
    if (!D.Parent)
      OS.indent(2 * D.Depth + 2) << "nest: W*_weighted=" << fmtNum(D.NestWStar)
                                 << " best_uniform_config=" << D.NestConfig << " (ROB "
                                 << TM.Window[D.NestConfig].ROB << ")\n";
  }
}

// Documented in WindowDemandAnalysis.h.
PreservedAnalyses WindowDemandPrinterPass::run(Function &F, FunctionAnalysisManager &FAM) {
  if (F.isDeclaration())
    return PreservedAnalyses::all();
  FunctionDemand &R = FAM.getResult<WindowDemandAnalysis>(F);
  R.print(OS, getTargetModel());
  return PreservedAnalyses::all();
}

} // namespace winhint
