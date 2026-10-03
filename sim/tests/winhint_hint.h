/**
 * @file
 * @brief RISC-V WinHint hint macros for the sim test programs.
 *
 * winhint_hint.h — RISC-V WinHint hint macros (docs/interfaces.md §2).
 * ori x0, x0, IMM with IMM = (payload << 5) | tag; a no-op on any RISC-V core.
 * Arguments must be compile-time constants. Define WINHINT_NO_HINTS to drop them.
 * On non-RISC-V targets the macros expand to nothing.
 */
#ifndef WINHINT_HINT_H
#define WINHINT_HINT_H
/** IMM[4:0] tag of setwin (0b10101). */
#define WINHINT_TAG_SETWIN 21
/** IMM[4:0] tag of region (0b10111). */
#define WINHINT_TAG_REGION 23
#if defined(__riscv) && !defined(WINHINT_NO_HINTS)
/** Emit `ori x0, x0, imm` (with a memory clobber, so it is not moved
 *  across memory accesses). */
#define WINHINT_INSN(imm) __asm__ volatile(".insn i 0x13, 6, x0, x0, %0" :: "i"(imm) : "memory")
#else
#define WINHINT_INSN(imm) do { } while (0)
#endif
/** setwin(w): w in entries, a multiple of 8 in 0..504 (0 = release). */
#define SETWIN(w)  WINHINT_INSN((((w) / 8) << 5) | WINHINT_TAG_SETWIN)
/** region(id): id in 0..63. */
#define REGION(id) WINHINT_INSN(((id) << 5) | WINHINT_TAG_REGION)
#endif
