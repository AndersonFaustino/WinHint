; Input of jones-paths.test (B6 JonesIQ): loops with independent iterations
; whose per-iteration chain goes through an integer multiply, an FP multiply,
; llvm.sqrt or an external call (latencies taken from the machine
; description), a loop without a preheader, and a loop without any
; queue-occupying instruction.
source_filename = "jones_paths.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

@I = global [4096 x i64] zeroinitializer
@F = global [4096 x float] zeroinitializer
@IO = global [4096 x i64] zeroinitializer
@FO = global [4096 x float] zeroinitializer

; IO[i] = I[i]^3 (integer multiplies).
define void @muli() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i64, ptr @I, i64 %i
  %a = load i64, ptr %p
  %m = mul i64 %a, %a
  %x.next = mul i64 %m, %a
  %q = getelementptr inbounds i64, ptr @IO, i64 %i
  store i64 %x.next, ptr %q
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 4096
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; FO[i] = F[i]^3 (FP multiplies).
define void @mulf() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds float, ptr @F, i64 %i
  %a = load float, ptr %p
  %m = fmul float %a, %a
  %x.next = fmul float %m, %a
  %q = getelementptr inbounds float, ptr @FO, i64 %i
  store float %x.next, ptr %q
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 4096
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; FO[i] = sqrt(sqrt(F[i])).
define void @sqrtf() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds float, ptr @F, i64 %i
  %a = load float, ptr %p
  %m = call float @llvm.sqrt.f32(float %a)
  %x.next = call float @llvm.sqrt.f32(float %m)
  %q = getelementptr inbounds float, ptr @FO, i64 %i
  store float %x.next, ptr %q
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 4096
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; IO[i] = ext(ext(I[i])) (external calls).
define void @callx() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i64, ptr @I, i64 %i
  %a = load i64, ptr %p
  %m = call i64 @ext(i64 %a)
  %x.next = call i64 @ext(i64 %m)
  %q = getelementptr inbounds i64, ptr @IO, i64 %i
  store i64 %x.next, ptr %q
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 4096
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; The loop header has two predecessors outside the loop: no preheader.
define i64 @nopreheader(i1 %b) {
entry:
  br i1 %b, label %loop, label %other
other:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ 7, %other ], [ %i.next, %loop ]
  %x = phi i64 [ 1, %entry ], [ 1, %other ], [ %x.next, %loop ]
  %p = getelementptr inbounds i64, ptr @I, i64 %i
  %a = load i64, ptr %p
  %x.next = add i64 %x, %a
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 4096
  br i1 %c, label %loop, label %exit
exit:
  ret i64 %x.next
}

; A loop of only an unconditional branch: nothing to schedule, no region.
define void @spin() {
entry:
  br label %loop
loop:
  br label %loop
}

declare i64 @ext(i64)
declare float @llvm.sqrt.f32(float)
