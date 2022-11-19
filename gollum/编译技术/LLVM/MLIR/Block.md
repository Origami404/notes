块是[操作](Operation.md)的列表. 它能接受参数, 它的 "值" 由终结操作定义. 借助块参数, 我们能实现 SSA 性质而不需要引入维护起来相对麻烦的 phi 指令.

语法:

````
block           ::= block-label operation+
block-label     ::= block-id block-arg-list? `:`
block-id        ::= caret-id
caret-id        ::= `^` suffix-id
value-id-and-type ::= value-id `:` type

// Non-empty list of names and types.
value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

block-arg-list ::= `(` value-id-and-type-list? `)`
````

例子:

````mlir
func.func @simple(i64, i1) -> i64 {
// 一个叫 ^bb0 的块
// 接受两个参数, 一个为类型 i64 的叫 a 的参数, 另一个为 i1 的叫 cond 的参数
// 一个函数的入口块的参数就是函数的参数
^bb0(%a: i64, %cond: i1):
  // 块内可以引用块的参数和其它块, 块也是值
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // 块能作为值的前提是给够了参数
  cf.br ^bb3(%a: i64)    
  
^bb2:
  %b = arith.addi %a, %a : i64
  cf.br ^bb3(%b: i64)    // Branch passes %b as the argument

^bb3(%c: i64):
  // 可以直接引用支配路径上的值 (%a)
  cf.br ^bb4(%c, %a : i64, i64)

^bb4(%d : i64, %e : i64):
  %0 = arith.addi %d, %e : i64
  return %0 : i64   // Return is also a terminator.
}
````

Blocks are also a fundamental concept that cannot be represented by operations because values defined in an operation cannot be accessed outside the operation.

块可以认为是操作的容器, 这本身不可以被某种 "block" 操作替代, 因为块中存放的值是可以被其它操作引用的 (只要在支配路径上), 而操作中定义的值不行.
