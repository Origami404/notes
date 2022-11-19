Graph regions are appropriate for concurrent semantics without control flow, or for modeling generic directed graph data structures.

在这种 region 里, 没有控制流的概念, 一般用于表示并发操作或者单纯的只是对图状数据结构的建模.

In graph regions, MLIR operations naturally represent nodes, while each MLIR value represents a multi-edge connecting a single source node and multiple destination nodes.

每个操作一般就相当于 "节点", 每个值就是一个单入边多出边的节点.

 > 
 > 所以操作跟值的关系到底是什么啊 (

````mlir
"test.graph_region"() ({ // A Graph region
  %1 = "op1"(%1, %3) : (i32, i32) -> (i32)  // OK: %1, %3 allowed here
  %2 = "test.ssacfg_region"() ({
     %5 = "op2"(%1, %2, %3, %4) : (i32, i32, i32, i32) -> (i32) // OK: %1, %2, %3, %4 all defined in the containing region
  }) : () -> (i32)
  %3 = "op2"(%1, %4) : (i32, i32) -> (i32)  // OK: %4 allowed here
  %4 = "op3"(%1) : (i32) -> (i32)
}) : () -> ()
````
