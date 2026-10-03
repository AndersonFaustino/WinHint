; Input of demand-paths.test (print<winhint-demand>): IR shapes that clang
; -O2 rarely produces but the window-demand analysis must handle (inline
; asm and atomics in a loop body, loop-invariant and symbolic-stride
; accesses, call-site argument bounds through phi/casts/arithmetic, local
; arrays, indirect accesses through a non-header phi).
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

@A = global [1048576 x float] zeroinitializer
@Idx = global [1048576 x i32] zeroinitializer
@G = global float 0.0
@Cnt = global i64 0

; Inline asm and an atomic RMW in the body: 1 instruction each.
define void @asm_atomic(i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  call void asm sideeffect "nop", ""()
  %old = atomicrmw add ptr @Cnt, i64 1 monotonic
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 1000
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; A load of the same scalar every iteration (not hoisted: no LICM here).
define float @invariant() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s.next, %loop ]
  %g = load float, ptr @G
  %s.next = fadd float %s, %g
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 1000
  br i1 %c, label %loop, label %exit
exit:
  ret float %s.next
}

; A[i * s] with an unknown s (no callers): the stride is assumed large and
; the loop is conservative.
define float @symbolic_stride(i64 %s) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %acc.next, %loop ]
  %k = mul i64 %i, %s
  %p = getelementptr inbounds float, ptr @A, i64 %k
  %x = load float, ptr %p
  %acc.next = fadd float %acc, %x
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 1000
  br i1 %c, label %loop, label %exit
exit:
  ret float %acc.next
}

; A[i * s] where the only call site passes s = 0: a zero stride (no misses).
define internal float @zero_stride(i64 %s) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %acc.next, %loop ]
  %k = mul i64 %i, %s
  %p = getelementptr inbounds float, ptr @A, i64 %k
  %x = load float, ptr %p
  %acc.next = fadd float %acc, %x
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 1000
  br i1 %c, label %loop, label %exit
exit:
  ret float %acc.next
}

define float @zero_stride_caller() {
  %r = call float @zero_stride(i64 0)
  ret float %r
}

; Trip count n bounded by the call sites: n in {(3 - 1) * 10 + 2,
; (5 - 1) * 10 + 2} = {22, 42} through phi, trunc, zext, sub, mul and add.
define internal void @tripc(i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds float, ptr @A, i64 %i
  store float 1.0, ptr %p
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @tripc_caller(i1 %b) {
entry:
  br i1 %b, label %a, label %c
a:
  br label %m
c:
  br label %m
m:
  %v = phi i64 [ 3, %a ], [ 5, %c ]
  %t = trunc i64 %v to i32
  %z = zext i32 %t to i64
  %s = sub i64 %z, 1
  %x = mul i64 %s, 10
  %y = add i64 %x, 2
  call void @tripc(i64 %y)
  ret void
}

; A call site with a non-constant argument: no bound, the trip is assumed.
define internal void @tripu(i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds float, ptr @A, i64 %i
  store float 2.0, ptr %p
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @tripu_caller(i64 %k) {
  %n = add i64 %k, 1
  call void @tripu(i64 %n)
  ret void
}

; 64 elements of a 32 MiB local array: the object does not fit any cache,
; so the first touch of each line misses to memory.
define float @local_array() {
entry:
  %buf = alloca [8388608 x float]
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds float, ptr %buf, i64 %i
  %x = load float, ptr %p
  %acc.next = fadd float %acc, %x
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 64
  br i1 %c, label %loop, label %exit
exit:
  ret float %acc.next
}

; A[Idx[i]] on even i only: the index reaches the address through a phi in
; the latch block (not the header), an indirect (not chase) access.
define float @indirect_merge() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %acc = phi float [ 0.0, %entry ], [ %acc.next, %latch ]
  %odd = and i64 %i, 1
  %even = icmp eq i64 %odd, 0
  br i1 %even, label %ld, label %latch
ld:
  %pi = getelementptr inbounds i32, ptr @Idx, i64 %i
  %ix = load i32, ptr %pi
  br label %latch
latch:
  %j = phi i32 [ %ix, %ld ], [ 0, %loop ]
  %je = zext i32 %j to i64
  %pa = getelementptr inbounds float, ptr @A, i64 %je
  %x = load float, ptr %pa
  %acc.next = fadd float %acc, %x
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 100000
  br i1 %c, label %loop, label %exit
exit:
  ret float %acc.next
}

; A[i] and A[i + d] with an unknown d: same base and stride but no constant
; distance, so two separate access groups.
define float @two_groups(i64 %d) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %acc.next, %loop ]
  %p0 = getelementptr inbounds float, ptr @A, i64 %i
  %x0 = load float, ptr %p0
  %id = add i64 %i, %d
  %p1 = getelementptr inbounds float, ptr @A, i64 %id
  %x1 = load float, ptr %p1
  %s = fadd float %x0, %x1
  %acc.next = fadd float %acc, %s
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 1000
  br i1 %c, label %loop, label %exit
exit:
  ret float %acc.next
}
