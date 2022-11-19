This dialect maps [LLVM IR](https://llvm.org/docs/LangRef.html) into MLIR by defining the corresponding operations and types. LLVM IR metadata is usually represented as MLIR attributes, which offer additional structure verification.

基本上就是直接一对一地将 LLVM IR 中的指令与概念在 MLIR 里写了一遍. 由 LLVM 方言到 LLVM IR 的翻译应该是显然的.

显著不同的是, 由于 MLIR 的设计特性, LLVM 方言里的 SSA 特性采用基本块参数而不是 phi 指令实现. 可以参考 [这里](https://mlir.llvm.org/docs/Dialects/LLVM/#phi-nodes-and-block-arguments)

## 将其它方言翻译到 LLVM 方言

然而, 将其它方言翻译到 LLVM 方言并不是那么地显然. 下面是一个翻译的例子:

````mlir
// from: 

// 这个文件就叫 memref-reshape.mlir
func.func @memref_reshape(%input : memref<2x3xf32>, %shape : memref<?xindex>) {
  %output = memref.reshape %input(%shape)
                : (memref<2x3xf32>, memref<?xindex>) -> memref<*xf32>
  return
}
````

随后, 我们将其转换为 LLVM 方言:

````bash
mlir-opt -convert-func-to-llvm -convert-memref-to-llvm -reconcile-unrealized-casts memref-reshape.mlir > memref-reshape.llvm.mlir
````

在这里我们使用了 `mlir-opt` 工具, 它是调用 mlir 各个 pass 的命令行工具, 可以通过 `mlir-opt -help` 查看都有些啥 pass (非常多). 我们主要关心下面的 pass:

* `-convert-xxxx-to-llvm`: 将 `xxxx` 方言翻译到 `llvm` 方言的 pass, 这些 pass 一般都是 `xxxx` 方言自己提供的.
* `-reconcile-unrealized-casts`: 将 `builtin.unrealized_conversion_cast` 操作剔除掉, 这些操作是为了使各个方言之间的转换顺序无关而插入的.
  * 关于这个参数的来源, 可以参考 [论坛中的这个帖子](https://discourse.llvm.org/t/how-to-convert-affine-dialect-llvm-ir/4710/3)

注意 pass 自然是按顺序执行的, 从左到右, 所以不要把 `-reconcile-unrealized-casts` 写到最前面去了.

然后, 我们就可以将 llvm 方言转换成 llvm ir:

````bash
mlir-translate -mlir-to-llvmir memref-reshape.llvm.mlir > memref-reshape.llvm
````

然后我们应该就能得到这个东西了:

````llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @memref_reshape(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11) {
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, ptr %1, 1
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 %2, 2
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 %3, 3, 0
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 %5, 4, 0
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 %4, 3, 1
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %6, 4, 1
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %7, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, ptr %8, 1
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 %9, 2
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %10, 3, 0
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 %11, 4, 0
  %25 = insertvalue { i64, ptr } undef, i64 %10, 0
  %26 = mul i64 2, %10
  %27 = add i64 %26, 1
  %28 = mul i64 %27, 8
  %29 = add i64 16, %28
  %30 = alloca i8, i64 %29, align 1
  %31 = insertvalue { i64, ptr } %25, ptr %30, 1
  store ptr %0, ptr %30, align 8
  %32 = getelementptr ptr, ptr %30, i32 1
  store ptr %1, ptr %32, align 8
  %33 = getelementptr ptr, ptr %30, i32 2
  store i64 %2, ptr %33, align 4
  %34 = getelementptr { ptr, ptr, i64, i64 }, ptr %30, i32 0, i32 3
  %35 = getelementptr i64, ptr %34, i64 %10
  %36 = sub i64 %10, 1
  br label %37

37:                                               ; preds = %41, %12
  %38 = phi i64 [ %47, %41 ], [ %36, %12 ]
  %39 = phi i64 [ %46, %41 ], [ 1, %12 ]
  %40 = icmp sge i64 %38, 0
  br i1 %40, label %41, label %48

41:                                               ; preds = %37
  %42 = getelementptr i64, ptr %8, i64 %38
  %43 = load i64, ptr %42, align 4
  %44 = getelementptr i64, ptr %34, i64 %38
  store i64 %43, ptr %44, align 4
  %45 = getelementptr i64, ptr %35, i64 %38
  store i64 %39, ptr %45, align 4
  %46 = mul i64 %39, %43
  %47 = sub i64 %38, 1
  br label %37

48:                                               ; preds = %37
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
````
