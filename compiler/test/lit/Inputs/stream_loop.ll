; Input of target-model.test: one streaming loop (B[i] = A[i] + 1 over 16 MiB
; arrays, independent DRAM misses) for checking parsed machine descriptions.
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

@A = global [4194304 x float] zeroinitializer
@B = global [4194304 x float] zeroinitializer

define void @stream() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pa = getelementptr inbounds float, ptr @A, i64 %i
  %x = load float, ptr %pa
  %y = fadd float %x, 1.0
  %pb = getelementptr inbounds float, ptr @B, i64 %i
  store float %y, ptr %pb
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, 4194304
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
