A region is an ordered list of MLIR [Block](Block.md)s. The semantics within a region is **not** imposed by the IR. Instead, the containing operation defines the semantics of the regions it contains. MLIR currently defines two kinds of regions: [SSACFG Region](SSACFG%20Region.md), which describe control flow between blocks, and [Graph Region](Graph%20Region.md), which do not require control flow between block. The kinds of regions within an operation are described using the [RegionKindInterface](https://mlir.llvm.org/docs/Interfaces/#regionkindinterfaces).

Regions do not have a name or an address, only the blocks contained in a region do. Regions must be contained within operations and have no type or attributes. The first block in the region is a special block called the ‘entry block’. The arguments to the entry block are also the arguments of the region itself.

区域是块的有序列表, 除此之外别无他物. MLIR 没有为区域预定义任何意义, 但区域有几个常用的子类 (其实是使用代表种类的枚举实现的): 代表控制流的 [SSACFG 区域](SSACFG%20Region.md), 以及单纯只是块集合的 [Graph 区域](Graph%20Region.md).

区域不会被绑定上任何的 "[名字](Identifier.md)" 或是地址, 只有块才能有名字/地址. 区域不能单独出现在 MLIR 顶层, 只能作为操作的参数. 

区域的第一个块成为区域的起始块.

**A function body is an example of a region**: it consists of a CFG of blocks and has additional semantic restrictions that other types of regions may not have.

函数体是区域的一种 (常见的, SSACFG 的) 特例.

Regions provide **hierarchical encapsulation** of programs: it is impossible to reference, i.e. branch to, a block which is not in the same region as the source of the reference, i.e. a terminator operation. Similarly, regions provides a natural scoping for value visibility: values defined in a region don’t escape to the enclosing region, if any.

区域为 MLIR 引入了树状的名称封装机制. 类似于绝大部分语言中的作用域的概念, 区域内部定义的名称只能被区域内部的操作引用, 或被区域内部的任意深儿子区域里的操作引用.

By default, operations inside a region can reference values defined outside of the region **whenever it would have been legal for operands of the enclosing operation to reference those values**, but this can be restricted using traits, such as [OpTrait::IsolatedFromAbove](https://mlir.llvm.org/docs/Traits/#isolatedfromabove), or a custom verifier.

需要注意的是, 在区域里引用父区域的名字的话, 一些常见的对名字误用的检查会默认关闭. (猜测可能是性能原因?) 不过你还能使用 verifier 来特别地检查这些名字, 或者对参数使用特定的 trait 来检查.

````mlir
  "any_op"(%a) ({ // if %a is in-scope in the containing region...
     // then %a is in-scope here too.
    %new_value = "another_op"(%a) : (i64) -> (i64)
  }) : (i64) -> (i64)
````
