In MLIR, control flow semantics of a region is indicated by [RegionKind::SSACFG](https://mlir.llvm.org/docs/Interfaces/#regionkindinterfaces). Informally, these regions support semantics where operations in a region ‘execute sequentially’. 

The determination of the next instruction to execute is the ‘passing of control flow’.

SSACFG 区域是用于代表控制流的区域, 在其中的块多半以某种顺序执行, 模拟出 "控制流传递" 的效果. 函数体就是一种 SSACFG Region.

````mlir
func.func @accelerator_compute(i64, i1) -> i64 { // An SSACFG region
^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // This def for %value does not dominate ^bb2
  %value = "op.convert"(%a) : (i64) -> i64
  cf.br ^bb3(%a: i64)    // Branch passes %a as the argument

^bb2:
  accelerator.launch() { // An SSACFG region
    ^bb0:
      // Region of code nested under "accelerator.launch", it can reference %a but
      // not %value.
      %new_value = "accelerator.do_something"(%a) : (i64) -> ()
  }
  // %new_value cannot be referenced outside of the region

^bb3:
  ...
}
````

一个操作里当然可以使用多个区域作为参数, 这时候控制流的传递完全由操作本身决定.

Regions allow defining an operation that creates a closure, for example by “boxing” the body of the region into a value they produce. It remains up to the operation to define its semantics. Note that if an operation triggers asynchronous execution of the region, it is under the responsibility of the operation caller to wait for the region to be executed guaranteeing that any directly used values remain live.

这种区域可以成为一种闭包, 引用上层区域的变量之后将自己作为一个值传递出去. 这种情况下当执行这个区域里的操作的时候, 如果碰到了引用的外部变量, 而这个变量恰好是一个异步操作的结果, 那么执行这个闭包的操作有责任保证这个结果是可用的. (既不是还没有, 也不是已经死了) 
