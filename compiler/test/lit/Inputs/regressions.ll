; Input of regressions.test: libm intrinsic summaries by element type and a
; call-site shl with negative shift amounts (argument bounds).
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

@D = global [512 x double] zeroinitializer
@F = global [512 x float] zeroinitializer

define void @exp_f64() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr @D, i64 %i
  %x = load double, ptr %p
  %y = call double @llvm.exp.f64(double %x)
  store double %y, ptr %p
  %i.next = add nuw nsw i64 %i, 2
  %c = icmp ult i64 %i.next, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @exp_v2f64() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr @D, i64 %i
  %x = load <2 x double>, ptr %p
  %y = call <2 x double> @llvm.exp.v2f64(<2 x double> %x)
  store <2 x double> %y, ptr %p
  %i.next = add nuw nsw i64 %i, 2
  %c = icmp ult i64 %i.next, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @exp_f32() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds float, ptr @F, i64 %i
  %x = load float, ptr %p
  %y = call float @llvm.exp.f32(float %x)
  store float %y, ptr %p
  %i.next = add nuw nsw i64 %i, 4
  %c = icmp ult i64 %i.next, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @exp_v4f32() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds float, ptr @F, i64 %i
  %x = load <4 x float>, ptr %p
  %y = call <4 x float> @llvm.exp.v4f32(<4 x float> %x)
  store <4 x float> %y, ptr %p
  %i.next = add nuw nsw i64 %i, 4
  %c = icmp ult i64 %i.next, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; Shift by a negative amount at the call site of a loop bounded by its argument.
define void @bounded(i64 %n) noinline {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr @D, i64 %i
  store double 0.0, ptr %p
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @caller(i1 %b) {
entry:
  %x = select i1 %b, i64 5, i64 3
  %y = select i1 %b, i64 -62, i64 -61
  %n = shl i64 %x, %y
  call void @bounded(i64 %n)
  ret void
}

declare double @llvm.exp.f64(double)
declare <2 x double> @llvm.exp.v2f64(<2 x double>)
declare float @llvm.exp.f32(float)
declare <4 x float> @llvm.exp.v4f32(<4 x float>)
