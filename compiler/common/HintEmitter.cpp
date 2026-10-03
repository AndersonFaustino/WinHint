//===- HintEmitter.cpp - WinHint ISA hint emission ------------------------===//
/**
 * @file
 * @brief Implementation of the hint encodings and of HintEmitter
 *        (all public functions are documented in HintEmitter.h).
 */
#include "HintEmitter.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace winhint {

unsigned riscvTag(HintKind K) {
  switch (K) {
  case HintKind::SetWin:
    return 21; // 0b10101
  case HintKind::Region:
    return 23; // 0b10111
  case HintKind::SetIQ:
    return 25; // 0b11001 (proposed, not in the contract)
  }
  return 21;
}

unsigned hintPayload(HintKind K, unsigned Value) {
  unsigned P = Value;
  if (K == HintKind::SetWin || K == HintKind::SetIQ)
    P = (Value + 7) / 8; // W = payload * 8, round up so the hint never undersizes
  return P > 63 ? 63 : P;
}

int riscvImm(HintKind K, unsigned Value) {
  return (int)((hintPayload(K, Value) << 5) | riscvTag(K)); // <= 2041, fits simm12
}

uint32_t x86Disp(HintKind K, unsigned Value) {
  return 0x57480000u | ((unsigned)K << 12) | hintPayload(K, Value);
}

uint32_t riscvWord(HintKind K, unsigned Value) {
  return 0x00006013u | ((uint32_t)riscvImm(K, Value) << 20);
}

bool parseEmitMode(StringRef S, EmitMode &M) {
  if (S == "asm")
    M = EmitMode::Asm;
  else if (S == "call")
    M = EmitMode::Call;
  else if (S == "none")
    M = EmitMode::None;
  else
    return false;
  return true;
}

const char *emitModeName(EmitMode M) {
  switch (M) {
  case EmitMode::Asm:
    return "asm";
  case EmitMode::Call:
    return "call";
  case EmitMode::None:
    return "none";
  }
  return "?";
}

HintEmitter::HintEmitter(Module &M, EmitMode Mode) : M(M), Mode(Mode) {
  Triple T(M.getTargetTriple());
  IsRISCV = T.isRISCV64() || T.isRISCV32();
  IsX86 = T.getArch() == Triple::x86_64;
}

std::string HintEmitter::describe(HintKind K, unsigned Value) const {
  std::string S;
  raw_string_ostream OS(S);
  const char *N = K == HintKind::SetWin ? "setwin" : K == HintKind::Region ? "region" : "setiq";
  OS << N << "(" << Value << ")";
  if (Mode == EmitMode::Asm && IsRISCV)
    OS << " ori x0,x0," << riscvImm(K, Value) << " [" << format_hex(riscvWord(K, Value), 10)
       << "]";
  else if (Mode == EmitMode::Asm && IsX86)
    OS << " nopl " << format_hex(x86Disp(K, Value), 10) << "(%rax)";
  return S;
}

// Documented in HintEmitter.h.
Instruction *HintEmitter::emit(Instruction *InsertPt, HintKind K, unsigned Value) {
  if (Mode == EmitMode::None)
    return nullptr;
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> B(InsertPt);
  CallInst *CI = nullptr;
  if (Mode == EmitMode::Call) {
    if (K == HintKind::SetIQ)
      return nullptr; // no runtime entry point for the proposed IQ-only hint
    const char *Fn = K == HintKind::SetWin ? "__winhint_setwin" : "__winhint_region";
    FunctionCallee Callee =
        M.getOrInsertFunction(Fn, FunctionType::get(Type::getVoidTy(Ctx), {B.getInt32Ty()}, false));
    if (Function *F = dyn_cast<Function>(Callee.getCallee()))
      F->addFnAttr(Attribute::NoUnwind);
    CI = B.CreateCall(Callee, {B.getInt32(K == HintKind::SetWin ? hintPayload(K, Value) * 8 : Value)});
  } else {
    std::string AsmStr;
    raw_string_ostream OS(AsmStr);
    if (IsRISCV)
      OS << ".insn i 0x13, 6, x0, x0, " << riscvImm(K, Value);
    else if (IsX86)
      OS << "nopl " << format_hex(x86Disp(K, Value), 10) << "(%rax)";
    else
      return nullptr;
    InlineAsm *IA = InlineAsm::get(FunctionType::get(Type::getVoidTy(Ctx), false), OS.str(), "",
                                   /*hasSideEffects=*/true);
    CI = B.CreateCall(IA);
    CI->addFnAttr(Attribute::NoUnwind);
  }
  MDBuilder MDB(Ctx);
  CI->setMetadata("winhint.hint",
                  MDNode::get(Ctx, {MDB.createConstant(B.getInt32((unsigned)K)),
                                    MDB.createConstant(B.getInt32(Value))}));
  return CI;
}

} // namespace winhint
