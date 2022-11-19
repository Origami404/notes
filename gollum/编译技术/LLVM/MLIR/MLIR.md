
 > 
 > MLIR is fundamentally based on a graph-like data structure of nodes, called [Operation](Operation.md)s, and edges, called *Values*. Each Value is the result of exactly one Operation or Block Argument, and has a *Value [Type](Type.md)* defined by the type system. [Operation](Operation.md)s are contained in [Block](Block.md)s and Blocks are contained in [Region](Region.md)s. Operations are also ordered within their containing block and Blocks are ordered in their containing region, although this order may or may not be semantically meaningful in a given kind of region. Operations may also contain regions, enabling hierarchical structures to be represented.

MLIR 里的 IR 是一个图状结构, [操作](Operation.md)是图中的点, 值是图中的边. 值可以是操作的结果, 或者是[块](Block.md)的参数, 并且带有一个[类型](Type.md). [区域](Region.md)是块的有序列表, 块是操作的有序列表, 而顺序的语义是由区域的种类决定的. 操作可以使用区域作为参数, 使得整个 IR 的结构具有递归性.

 > 
 > 　Operations can represent many different concepts, from higher-level concepts like function definitions, function calls, buffer allocations, view or slices of buffers, and process creation, to lower-level concepts like target-independent arithmetic, target-specific instructions, configuration registers, and logic gates. These different concepts are represented by different operations in MLIR and the set of operations usable in MLIR can be arbitrarily extended.

操作的外延是开放的, 它可以表示 *任何* 东西, 只要你去定义它. MLIR 里的操作是完全可扩展, 可自定义的.

 > 
 > MLIR also provides an extensible framework for transformations on operations, using familiar concepts of compiler [Passes](https://mlir.llvm.org/docs/Passes/). Enabling an arbitrary set of passes on an arbitrary set of operations results in a significant scaling challenge, since each transformation must potentially take into account the semantics of any operation. MLIR addresses this complexity by allowing operation semantics to be described abstractly using [Traits](https://mlir.llvm.org/docs/Traits/) and [Interfaces](https://mlir.llvm.org/docs/Interfaces/), enabling transformations to operate on operations more generically. Traits often describe verification constraints on valid IR, enabling complex invariants to be captured and checked. (see [Op vs Operation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations))

MLIR 为操作之间的转换提供了框架支持. 使用跟传统多趟编译器里类似的 Pass, 你可以在不同的操作之间转来转去. MLIR 还允许通过 Trait 与 Interface 来描述操作的特性, 使我们能编写出更加通用的转换 Pass. Trait 通常还用于 IR 的检验.

 > 
 > One obvious application of MLIR is to represent an [SSA-based](https://en.wikipedia.org/wiki/Static_single_assignment_form) IR, like the LLVM core IR, with appropriate choice of operation types to define Modules, Functions, Branches, Memory Allocation, and verification constraints to ensure the SSA Dominance property. MLIR includes a collection of dialects which defines just such structures. However, MLIR is intended to be general enough to represent other compiler-like data structures, such as Abstract Syntax Trees in a language frontend, generated instructions in a target-specific backend, or circuits in a High-Level Synthesis tool.

````mlir
// Compute A*B using an implementation of multiply kernel and print the
// result using a TensorFlow op. The dimensions of A and B are partially
// known. The shapes are assumed to match.
func.func @mul(%A: tensor<100x?xf32>, %B: tensor<?x50xf32>) -> (tensor<100x50xf32>) {
  // Compute the inner dimension of %A using the dim operation.
  %n = memref.dim %A, 1 : tensor<100x?xf32>

  // Allocate addressable "buffers" and copy tensors %A and %B into them.
  %A_m = memref.alloc(%n) : memref<100x?xf32>
  memref.tensor_store %A to %A_m : memref<100x?xf32>

  %B_m = memref.alloc(%n) : memref<?x50xf32>
  memref.tensor_store %B to %B_m : memref<?x50xf32>

  // Call function @multiply passing memrefs as arguments,
  // and getting returned the result of the multiplication.
  %C_m = call @multiply(%A_m, %B_m)
          : (memref<100x?xf32>, memref<?x50xf32>) -> (memref<100x50xf32>)

  memref.dealloc %A_m : memref<100x?xf32>
  memref.dealloc %B_m : memref<?x50xf32>

  // Load the buffer data into a higher level "tensor" value.
  %C = memref.tensor_load %C_m : memref<100x50xf32>
  memref.dealloc %C_m : memref<100x50xf32>

  // Call TensorFlow built-in function to print the result tensor.
  "tf.Print"(%C){message: "mul result"} : (tensor<100x50xf32>) -> (tensor<100x50xf32>)

  return %C : tensor<100x50xf32>
}

// A function that multiplies two memrefs and returns the result.
func.func @multiply(%A: memref<100x?xf32>, %B: memref<?x50xf32>)
          -> (memref<100x50xf32>)  {
  // Compute the inner dimension of %A.
  %n = memref.dim %A, 1 : memref<100x?xf32>

  // Allocate memory for the multiplication result.
  %C = memref.alloc() : memref<100x50xf32>

  // Multiplication loop nest.
  affine.for %i = 0 to 100 {
     affine.for %j = 0 to 50 {
        memref.store 0 to %C[%i, %j] : memref<100x50xf32>
        affine.for %k = 0 to %n {
           %a_v  = memref.load %A[%i, %k] : memref<100x?xf32>
           %b_v  = memref.load %B[%k, %j] : memref<?x50xf32>
           %prod = arith.mulf %a_v, %b_v : f32
           %c_v  = memref.load %C[%i, %j] : memref<100x50xf32>
           %sum  = arith.addf %c_v, %prod : f32
           memref.store %sum, %C[%i, %j] : memref<100x50xf32>
        }
     }
  }
  return %C : memref<100x50xf32>
}
````

MLIR 使用[方言](Dialect.md)的概念来进行语言的扩展. 官方附带了一些方言, 详见 [Dialect](Dialect.md). 如无特殊说明, 本系列中提到的各类二进制工具, 都可以在编译后的 llvm 项目里找到. 具体编译方法请参考 [Getting Started with Clang & LLVM](../Getting%20Started%20with%20Clang%20&%20LLVM.md), 但是记得在 `${llvm_enable_projects}` 里加入 mlir, 比如使用 "clang;mlir". 

将常见方言下降到 LLVM IR 的过程可以参考 [LLVM Dialect > 将其它方言翻译到 LLVM 方言](LLVM%20Dialect.md#jiang-qi-ta-fang-yan-fan-yi-dao-llvm-fang-yan).
