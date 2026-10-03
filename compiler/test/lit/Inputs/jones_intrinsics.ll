; Input of regressions.test (B6 JonesIQ): the same loop with integer and with FP
; intrinsics; empty source_filename (kernel name from the module identifier).
source_filename = ""
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

@A = global [4096 x i32] zeroinitializer
@G = global [4096 x float] zeroinitializer

define void @clampi() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i32, ptr @A, i64 %i
  %x = load i32, ptr %p
  %a = call i32 @llvm.smax.i32(i32 %x, i32 0)
  %b = call i32 @llvm.smin.i32(i32 %a, i32 255)
  %c = call i32 @llvm.umax.i32(i32 %b, i32 1)
  %d = call i32 @llvm.ctpop.i32(i32 %c)
  store i32 %d, ptr %p
  %i.next = add nuw nsw i64 %i, 1
  %e = icmp ult i64 %i.next, 4096
  br i1 %e, label %loop, label %exit
exit:
  ret void
}

define void @clampf() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds float, ptr @G, i64 %i
  %x = load float, ptr %p
  %a = call float @llvm.maxnum.f32(float %x, float 0.0)
  %b = call float @llvm.minnum.f32(float %a, float 255.0)
  %c = call float @llvm.maxnum.f32(float %b, float 1.0)
  %d = call float @llvm.fabs.f32(float %c)
  store float %d, ptr %p
  %i.next = add nuw nsw i64 %i, 1
  %e = icmp ult i64 %i.next, 4096
  br i1 %e, label %loop, label %exit
exit:
  ret void
}

declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.ctpop.i32(i32)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.fabs.f32(float)
