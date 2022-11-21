MLIR supports **multi-dimensional `vector`** types and custom operations on those types. A generic, retargetable, higher-order `vector` type (`n-D` with `n > 1`) is a structured type, that carries semantic information useful for transformations. This document discusses retargetable abstractions that exist in MLIR today and operate on SSA-values of type `vector` along with pattern rewrites and lowerings that enable targeting specific instructions on concrete targets. These abstractions serve to separate concerns between operations on `memref` (a.k.a buffers) and operations on `vector` values. 

MLIR 提供了一个通用的, 平台无关的 `vector` 类型, 这个类型可以是多维的. 借助此方言, 我们得以以 SSA 形式表示与优化任意的向量化操作, 同时逐步地引入平台信息, 底层化到具体目标上的 SIMD 指令集上.

## CodeGen Path

![CodeGen Level in MLIR.png](assert/CodeGen%20Level%20in%20MLIR.png)
![Dialects near MLIR.png](assert/Dialects%20near%20MLIR.png)

## 操作

### `vector.vscale`

搞不懂

### `vector.bitcast`

````
`vector.bitcast` $source attr-dict `:` type($source) `to` type($result)
````

按位转换. 不能动阶, 可以动基类型和维度, 总的按字节长度要保持前后一致.

````mlir
// Example casting to a smaller element type.
%1 = vector.bitcast %0 : vector<5x1x4x3xf32> to vector<5x1x4x6xi16>

// Example casting to a bigger element type.
%3 = vector.bitcast %2 : vector<10x12x8xi8> to vector<10x12x2xi32>

// Example casting to an element type of the same size.
%5 = vector.bitcast %4 : vector<5x1x4x3xf32> to vector<5x1x4x3xi32>

// Example casting of 0-D vectors.
%7 = vector.bitcast %6 : vector<f32> to vector<i32>
````
