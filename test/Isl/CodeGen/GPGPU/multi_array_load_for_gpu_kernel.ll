; REQUIRES: nvptx-registered-target
; RUN: opt %loadPolly %defaultOpts -polly-only-func=main -polly-codegen-isl -polly-dump-gpu-kernels -analyze < %s | FileCheck %s

; int A[1024];
; int B[1024];

; void initArray() {
;   int i,j;
;   for(i = 0; i < 1024; i++) {
;     A[i] = 4;
;     B[i] = i+12;
;   }
; }

; int main() {
;   int i,j;
;
;   initArray();
;
; #pragma scop
;   for(i = 0; i < 5; i++)
;     A[i] = B[0]+5;
; #pragma endscop
;
;   return 0;
; }

; ModuleID = './multi_array_load_for_gpu_kernel.s'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1024 x i32] zeroinitializer, align 16
@B = common global [1024 x i32] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define void @initArray() #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr [1024 x i32]* @A, i64 0, i64 %indvar
  %0 = add i64 %indvar, 12
  %add = trunc i64 %0 to i32
  %arrayidx2 = getelementptr [1024 x i32]* @B, i64 0, i64 %indvar
  store i32 4, i32* %arrayidx, align 4
  store i32 %add, i32* %arrayidx2, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @initArray()
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr [1024 x i32]* @A, i64 0, i64 %indvar
  %0 = load i32* getelementptr inbounds ([1024 x i32]* @B, i32 0, i64 0), align 4
  %add = add nsw i32 %0, 5
  store i32 %add, i32* %arrayidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, 5
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret i32 0
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5.0 "}

; CHECK-DAG:  %14 = bitcast i8* %ptx.Array.MemRef_A to i32*
; CHECK-DAG:  %17 = getelementptr i32* %14, i64 %13
; CHECK-DAG:  store i32 %p_add, i32* %17
