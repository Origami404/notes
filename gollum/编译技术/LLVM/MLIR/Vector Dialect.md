MLIR supports **multi-dimensional `vector`** types and custom operations on those types. A generic, retargetable, higher-order `vector` type (`n-D` with `n > 1`) is a structured type, that carries semantic information useful for transformations. This document discusses retargetable abstractions that exist in MLIR today and operate on SSA-values of type `vector` along with pattern rewrites and lowerings that enable targeting specific instructions on concrete targets. These abstractions serve to separate concerns between operations on `memref` (a.k.a buffers) and operations on `vector` values. 

MLIR 提供了一个通用的, 平台无关的 `vector` 类型, 这个类型可以是多维的. 借助此方言, 我们得以以 SSA 形式表示与优化任意的向量化操作, 同时逐步地引入平台信息, 底层化到具体目标上的 SIMD 指令集上.

## CodeGen Path

![CodeGen Level in MLIR.png](assert/CodeGen%20Level%20in%20MLIR.png)
![Dialects near MLIR.png](assert/Dialects%20near%20MLIR.png)
