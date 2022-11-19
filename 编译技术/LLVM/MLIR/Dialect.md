Dialects are the mechanism by which to engage with and extend the MLIR ecosystem. They allow for defining new [operations](https://mlir.llvm.org/docs/LangRef/#operations), as well as [attributes](https://mlir.llvm.org/docs/LangRef/#attributes) and [types](https://mlir.llvm.org/docs/LangRef/#type-system).

Dialects provide a grouping mechanism for abstraction under a unique `namespace`.

Dialects are produced and consumed by certain passes. MLIR provides a [framework](https://mlir.llvm.org/docs/DialectConversion/) to convert between, and within, different dialects.

方言是 MLIR 的基本组成模块. 一个方言基本上可以完全定义属于它的任何东西. 首先, 方言会分到一个命名空间 (比如向量化方言就用 vector, 仿射分析方言就用 affine). 随后它就可以使用该命名空间下的任意标识符. 它可以定义新的操作/属性/类型/..., 甚至还可以定义属于它自己的文本表示形式的 parser 和 dumper 以方便开发. 随后它可以通过 pass 被变换成同一种方言的不同形式, 也可以直接变换到不同的方言去. 这个变换既可以是完全的 (类似传统分趟编译器一样), 又可以是部分的 (只将本方言的特定指令转换到其它方言去).

MLIR 目前官方支持的方言有:

* [Builtin Dialect](Builtin%20Dialect.md)
* [MemoryRef Dialect](MemoryRef%20Dialect.md)
* [LLVM Dialect](LLVM%20Dialect.md)
* [Vector Dialect](Vector%20Dialect.md)
* [Affine Dialect](Affine%20Dialect.md)
* [Func dialect](https://mlir.llvm.org/docs/Dialects/Func/)
* [GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
* [SPIR-V dialect](https://mlir.llvm.org/docs/Dialects/SPIR-V/)
* [Vector dialect](https://mlir.llvm.org/docs/Dialects/Vector/)

其中个人最重要的是 LLVM 方言. LLVM 方言提供了一个到 LLVM IR 的 pass, 其它方言只要能转换成 LLVM 方言, 就可以获得编译成可执行文件 (当然, 还有享受 LLVM 优化) 的能力. 当然, 它最主要的用途是拿来观察各个方言里操作的语义, 因为我能看懂 LLVM IR :)
