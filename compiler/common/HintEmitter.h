//===- HintEmitter.h - WinHint ISA hint emission ----------------*- C++ -*-===//
//
// Emits the hint contract of docs/interfaces.md §2:
//   RISC-V : ori x0, x0, IMM   via  .insn i 0x13, 6, x0, x0, IMM
//            IMM = (payload << 5) | tag, tag 21 = setwin (W = payload*8),
//            tag 23 = region (id = payload)
//   x86-64 : nopl DISP32(%rax), DISP32 = 0x57480000 | (kind << 12) | payload
//   call   : __winhint_setwin(W) / __winhint_region(id)
//
// Shared with compiler/baselines (B6 JonesIQ): HintEmitter(M, Mode).emit(),
// describe(), parseEmitMode(), HintKind and the encoding helpers are a stable
// API (additive changes only). Every emitted hint is an `asm sideeffect`
// call (or a runtime call) tagged with !winhint.hint !{i32 kind, i32 value}.
//
// HintKind::SetIQ is NOT part of the contract. It is a *proposed* encoding for
// the B6 "IQ only as published" variant (tag 25 / x86 kind 3, payload = IQ/8)
// and is only emitted when explicitly requested (-jones-iq-encoding=setiq).
//
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Encoding helpers and IR emitter for the WinHint hint ISA
 *        (setwin / region, docs/interfaces.md §2).
 *
 * Hints are inserted either as side-effecting inline asm (RISC-V HINT-space
 * ORI or x86-64 multi-byte NOP, chosen from the module triple) or as calls to
 * the libwinhint runtime (__winhint_setwin / __winhint_region).
 */
#ifndef WINHINT_HINTEMITTER_H
#define WINHINT_HINTEMITTER_H

#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>
#include <string>

namespace winhint {

/// How hints are materialized (-winhint-emit=asm|call|none).
enum class EmitMode {
  Asm,  ///< ISA hint as inline asm (RISC-V ORI or x86-64 NOP, from the triple)
  Call, ///< call to __winhint_setwin / __winhint_region (libwinhint)
  None  ///< emit nothing
};
/// Hint kind; the numeric value is the `kind` code of docs/interfaces.md §2.
enum class HintKind : unsigned {
  SetWin = 1, ///< setwin(W): advisory window size in entries (0 = release)
  Region = 2, ///< region(id): entry into static region id
  SetIQ = 3   ///< proposed IQ-only hint for B6; not part of the contract
};

/// @brief RISC-V tag (imm[4:0]) per kind: 21 setwin, 23 region, 25 setiq.
unsigned riscvTag(HintKind K);
/// @brief Payload for a kind: setwin/setiq carry ceil(W/8), region the id;
///        clamped to 63.
unsigned hintPayload(HintKind K, unsigned Value);
/// @brief 12-bit ORI immediate for RISC-V: (payload << 5) | tag.
int riscvImm(HintKind K, unsigned Value);
/// @brief 32-bit displacement of the x86 NOP: 0x57480000 | (kind << 12) | payload.
uint32_t x86Disp(HintKind K, unsigned Value);
/// @brief Raw RISC-V instruction word 0x00006013 | (imm << 20) (for tests).
uint32_t riscvWord(HintKind K, unsigned Value);

/// @brief Parse "asm", "call" or "none".
/// @param[in]  S Mode string.
/// @param[out] M Parsed mode; unchanged on failure.
/// @return false if S is not a known mode.
bool parseEmitMode(llvm::StringRef S, EmitMode &M);
/// @brief Inverse of parseEmitMode(): "asm", "call" or "none".
const char *emitModeName(EmitMode M);

/**
 * @brief Inserts setwin/region (and optionally setiq) hints into a module.
 *
 * Every emitted call carries `!winhint.hint !{i32 kind, i32 value}` metadata,
 * where value is the requested (unrounded) W or id.
 */
class HintEmitter {
public:
  /// @brief Bind to module M and detect RISC-V (32/64) or x86-64 from its triple.
  HintEmitter(llvm::Module &M, EmitMode Mode);

  /**
   * @brief Insert one hint before InsertPt.
   *
   * In call mode setwin passes the rounded-up window (payload * 8) and region
   * the id; setiq has no runtime entry point and is skipped.
   *
   * @param InsertPt Instruction before which the hint is inserted.
   * @param K        Hint kind.
   * @param Value    W in entries (setwin/setiq) or region id.
   * @return The created call, or nullptr (mode none, setiq in call mode, or
   *         unsupported target in asm mode).
   */
  llvm::Instruction *emit(llvm::Instruction *InsertPt, HintKind K, unsigned Value);

  /// @brief Emission mode given at construction.
  EmitMode mode() const { return Mode; }
  /// @brief Whether the module targets RISC-V (32 or 64 bit).
  bool isRISCV() const { return IsRISCV; }
  /// @brief Whether the module targets x86-64.
  bool isX86() const { return IsX86; }
  /// @brief Whether asm emission is supported for this module's triple.
  bool asmSupported() const { return IsRISCV || IsX86; }
  /// @brief Human-readable form, e.g. "setwin(128) ori x0,x0,533 [0x21506013]";
  ///        the encoding is appended only in asm mode on a supported target.
  std::string describe(HintKind K, unsigned Value) const;

private:
  llvm::Module &M;                     ///< module receiving the hints
  EmitMode Mode;                       ///< emission mode
  bool IsRISCV = false, IsX86 = false; ///< target ISA from the triple
};

} // namespace winhint

#endif
