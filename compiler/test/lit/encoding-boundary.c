/**
 * @file
 * @brief Lit test of the setwin/region encodings at boundary values
 *        (RISC-V and x86-64 asm, and runtime-call mode) via from-json mode.
 */
// Hint encodings at the boundary values (docs/interfaces.md §2), through the
// from-json mode: regions a0..a7 get region(id) + setwin(W) with
//   W = 0 (release), 1, 8, 9, 504 (largest payload), 505, 100000 (clamped to
//   payload 63 = 504), and a bare -1 (release).
// payload = ceil(W/8), max 63. RISC-V: word 0x00006013 | (((payload<<5)|tag) << 20),
// tag 21 setwin / 23 region. x86: 0F 1F 80 + disp32 0x57480000|(kind<<12)|payload.
// Both LLVM's and binutils' disassemblers must decode the hints as plain
// ori x0 / nopl (i.e. valid no-ops).
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: echo '{"0":{"W":0},"1":{"W":1},"2":{"W":8},"3":{"W":9},"4":{"W":504},"5":{"W":505},"6":{"W":100000},"7":-1}' > %t/map.json
// RUN: %rvcc -O2 -fno-unroll-loops -fno-vectorize %whflags -mllvm -winhint-mode=from-json=%t/map.json -mllvm -winhint-out-dir=%t -c %s -o %t/rv.o
// RUN: llvm-objdump -d %t/rv.o | FileCheck %s --check-prefix=RV
// RUN: %if riscv64-gnu-objdump %{ riscv64-conda-linux-gnu-objdump -d %t/rv.o | FileCheck %s --check-prefix=RVGNU %}
// RUN: %x86cc -O2 -fno-unroll-loops -fno-vectorize %whflags -mllvm -winhint-mode=from-json=%t/map.json -c %s -o %t/x86.o
// RUN: llvm-objdump -d %t/x86.o | FileCheck %s --check-prefix=X86
// RUN: %if x86_64-gnu-objdump %{ x86_64-conda-linux-gnu-objdump -d %t/x86.o | FileCheck %s --check-prefix=X86GNU %}
// RUN: %rvcc -O2 -fno-unroll-loops -fno-vectorize %whflags -mllvm -winhint-mode=from-json=%t/map.json -mllvm -winhint-emit=call -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CALL
// RUN: %python -c "import json; d=json.load(open('%t/encoding-boundary.regions.json')); assert d['num_regions'] == 8 and [d['regions'][str(i)]['function'] for i in range(8)] == ['a%%d' % i for i in range(8)], d"

/** @brief Region 0 (map W = 0 (release)): add 1 to n elements of p. */
void a0(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 1.0f; }
/** @brief Region 1 (map W = 1): add 2 to n elements of p. */
void a1(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 2.0f; }
/** @brief Region 2 (map W = 8): add 3 to n elements of p. */
void a2(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 3.0f; }
/** @brief Region 3 (map W = 9): add 4 to n elements of p. */
void a3(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 4.0f; }
/** @brief Region 4 (map W = 504): add 5 to n elements of p. */
void a4(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 5.0f; }
/** @brief Region 5 (map W = 505): add 6 to n elements of p. */
void a5(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 6.0f; }
/** @brief Region 6 (map W = 100000): add 7 to n elements of p. */
void a6(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 7.0f; }
/** @brief Region 7 (map W = -1 (release)): add 8 to n elements of p. */
void a7(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 8.0f; }

// region(id) precedes setwin(W) in each preheader (the scheduler may move
// other instructions between them).
// RV-LABEL: <a0>:
// RV:      01706013 {{.*}}ori zero, zero, 0x17
// RV:      01506013 {{.*}}ori zero, zero, 0x15
// RV-LABEL: <a1>:
// RV:      03706013 {{.*}}ori zero, zero, 0x37
// RV:      03506013 {{.*}}ori zero, zero, 0x35
// RV-LABEL: <a2>:
// RV:      05706013 {{.*}}ori zero, zero, 0x57
// RV:      03506013 {{.*}}ori zero, zero, 0x35
// RV-LABEL: <a3>:
// RV:      07706013 {{.*}}ori zero, zero, 0x77
// RV:      05506013 {{.*}}ori zero, zero, 0x55
// RV-LABEL: <a4>:
// RV:      09706013 {{.*}}ori zero, zero, 0x97
// RV:      7f506013 {{.*}}ori zero, zero, 0x7f5
// RV-LABEL: <a5>:
// RV:      0b706013 {{.*}}ori zero, zero, 0xb7
// RV:      7f506013 {{.*}}ori zero, zero, 0x7f5
// RV-LABEL: <a6>:
// RV:      0d706013 {{.*}}ori zero, zero, 0xd7
// RV:      7f506013 {{.*}}ori zero, zero, 0x7f5
// RV-LABEL: <a7>:
// RV:      0f706013 {{.*}}ori zero, zero, 0xf7
// RV:      01506013 {{.*}}ori zero, zero, 0x15

// RVGNU-LABEL: <a0>:
// RVGNU:      01706013 {{.*}}ori zero,zero,23
// RVGNU:      01506013 {{.*}}ori zero,zero,21
// RVGNU-LABEL: <a4>:
// RVGNU:      09706013 {{.*}}ori zero,zero,151
// RVGNU:      7f506013 {{.*}}ori zero,zero,2037
// RVGNU-LABEL: <a7>:
// RVGNU:      0f706013 {{.*}}ori zero,zero,247
// RVGNU:      01506013 {{.*}}ori zero,zero,21

// X86-LABEL: <a0>:
// X86:      0f 1f 80 00 20 48 57 {{.*}}nopl 0x57482000(%rax)
// X86:      0f 1f 80 00 10 48 57 {{.*}}nopl 0x57481000(%rax)
// X86-LABEL: <a1>:
// X86:      0f 1f 80 01 20 48 57 {{.*}}nopl 0x57482001(%rax)
// X86:      0f 1f 80 01 10 48 57 {{.*}}nopl 0x57481001(%rax)
// X86-LABEL: <a2>:
// X86:      0f 1f 80 02 20 48 57 {{.*}}nopl 0x57482002(%rax)
// X86:      0f 1f 80 01 10 48 57 {{.*}}nopl 0x57481001(%rax)
// X86-LABEL: <a3>:
// X86:      0f 1f 80 03 20 48 57 {{.*}}nopl 0x57482003(%rax)
// X86:      0f 1f 80 02 10 48 57 {{.*}}nopl 0x57481002(%rax)
// X86-LABEL: <a4>:
// X86:      0f 1f 80 04 20 48 57 {{.*}}nopl 0x57482004(%rax)
// X86:      0f 1f 80 3f 10 48 57 {{.*}}nopl 0x5748103f(%rax)
// X86-LABEL: <a5>:
// X86:      0f 1f 80 05 20 48 57 {{.*}}nopl 0x57482005(%rax)
// X86:      0f 1f 80 3f 10 48 57 {{.*}}nopl 0x5748103f(%rax)
// X86-LABEL: <a6>:
// X86:      0f 1f 80 06 20 48 57 {{.*}}nopl 0x57482006(%rax)
// X86:      0f 1f 80 3f 10 48 57 {{.*}}nopl 0x5748103f(%rax)
// X86-LABEL: <a7>:
// X86:      0f 1f 80 07 20 48 57 {{.*}}nopl 0x57482007(%rax)
// X86:      0f 1f 80 00 10 48 57 {{.*}}nopl 0x57481000(%rax)

// X86GNU-LABEL: <a0>:
// X86GNU:      0f 1f 80 00 20 48 57 {{.*}}nopl 0x57482000(%rax)
// X86GNU:      0f 1f 80 00 10 48 57 {{.*}}nopl 0x57481000(%rax)
// X86GNU-LABEL: <a4>:
// X86GNU:      0f 1f 80 04 20 48 57 {{.*}}nopl 0x57482004(%rax)
// X86GNU:      0f 1f 80 3f 10 48 57 {{.*}}nopl 0x5748103f(%rax)

// Runtime-call mode passes W = payload * 8 and the region id.
// CALL-LABEL: define {{.*}} @a0(
// CALL:      call void @__winhint_region(i32 0)
// CALL-NEXT: call void @__winhint_setwin(i32 0)
// CALL-LABEL: define {{.*}} @a1(
// CALL:      call void @__winhint_setwin(i32 8)
// CALL-LABEL: define {{.*}} @a3(
// CALL:      call void @__winhint_setwin(i32 16)
// CALL-LABEL: define {{.*}} @a4(
// CALL:      call void @__winhint_setwin(i32 504)
// CALL-LABEL: define {{.*}} @a6(
// CALL:      call void @__winhint_region(i32 6)
// CALL-NEXT: call void @__winhint_setwin(i32 504)
// CALL-LABEL: define {{.*}} @a7(
// CALL:      call void @__winhint_setwin(i32 0)
