Ref: <https://mlir.llvm.org/docs/Dialects/Affine/>

MLIR uses techniques from polyhedral compilation to make dependence analysis and loop transformations efficient and reliable. This section introduces some of the core concepts that are used throughout the document.

 > 
 > Affine: 仿射变换, 基本上指 $x \rightarrow a x + b$ 类型的变换, 能保证平行性.

本方言主要用于代替 "循环" 这种概念, 方便向量化的依赖分析和对循环的变换. 

## 原理

### 个人理解

目前**我的个人理解**是, 大部分真正干活的循环就是一个从 A 到 B 一直干事, 并且在一个线性的内存空间中按照某种看起来像是多维数组的方式访问/写入内存的东西. 而这完全可以被建模为:

* 有一个或多个索引 "变量" (术语叫 Dimensions, 维度), 相当于变换里的 $x$
* 有好多个表示偏移或者基地址的 "常量" (术语叫 Symbol, 符号), 相当于变换里的 $a$ 和 $b$.

于是实际上我们可以详细地将我们访问内存的方式, 它既不是直接下降到 "对线性地址空间的访问" 这么低的层次, 也不是高层到完全忽略 "地址是线性的, 我们有可能越界" 的情况. 而是可以根据需要, 同时地在我们的索引和底层的地址之间切换. 所以它在分析多层循环的数据依赖性的时候会很有用.

但是我们完全可以眼界放宽. 我们一定得是一个多维到一维的变换吗? 我完全可以不从真正的循环结构直接下降到线性内存, 而是先下降一半或者怎么样的到一个低维一点的数据, 也就是说, 它可以在两个不同的索引空间之间表达它们是要如何转换的.

同理, 它就一定得是只能处理内存访问吗? 我为什么不能将 "对给定位置的内存访问" 这件事抽象成一个任务或者是函数, 类似 `do_something(索引)` 这样子呢? 这样就能进一步抽象出循环本身, 而不是将分析局限在内存上. 

说到底, 循环本身事实上代表了 "我要遍历某某索引空间对每一个地方干一个什么事", 而对循环的优化的可能就在于, 这个索引空间实际上是 "假的", 它总是对应到某种更底层或更高层的索引空间去 (比如线性内存地址). 如果我们能根据遍历方式, 分析出它在目标索引空间的干活模式或者各个活的依赖性, 我们就有可能消除不必要的重复活或者做并行化. 而仿射变换这个概念, 正是用于表达这种抽象的, 索引空间的变换模式的东西.

举个例子, 如下图:

![Pasted image 20221119224224.png](assert/Pasted%20image%2020221119224224.png)

我们有一个 MemRef (小的那个), 它是另一块内存空间 (大的那个) 的一个局部视图. 我们可以使用下面的仿射变换来指定它的索引模式:

````
#imapB2A = (d0, d1)[S0, S1] -> (d0 + S0, d1 + S1)
````

其中 `#` 的意思是它是一个属性, `#imapB2A = ` 就是给我们现在在写的东西一个名字. 右边的意思是, 我们有两个 "变量" 充当索引, 习惯上命名为 `d0`, `d1`, ...; 有两个 "常量" (符号) 充当偏移量, 习惯上命名 `S0`, `S1`, .... 那么上面的式子实际上就表达了一个 "模式", 它能将一个索引空间变到另一个去.

 > 
 > 实际上完整写出来应该是这样的:
 > 
 > `#imapB2A = affine_map<(d0, d1)[S0, S1] -> (d0 + S0, d1 + S1)>`
 > 
 > 但是出于仿射变换的超然地位, 它有特殊的语法, 所以可以适当省略 `affine_map<>`.

那么当我们在分析的时候, 我们就可以直接从这个变换看出这两个 MemRef 的依赖关系和它们是如何对应的, 为循环分析做准备. 实际上仿射变换正是描述 MemRef 的布局模式的一种非常灵活的方式, 详见 [MemoryRef Dialect > 布局模式](MemoryRef%20Dialect.md#bu-ju-mo-shi).

## 属性

### MLIR 里的仿射变换

实际上, 循环里常用的模式不只有线性的 `ax+b`, 另外一个很常用的模式是取余来模拟循环数组之类的 (~~这是否也算一种内部的卷? 内卷!~~). 所以在 MLIR 的仿射方言里, "仿射变换" 实际上是指一个更广义的, 支持 (对字面量) 除法和取余的变换:

````
affine-expr ::= `(` affine-expr `)`
              | affine-expr `+` affine-expr
              | affine-expr `-` affine-expr
              | `-`? integer-literal `*` affine-expr
              | affine-expr `ceildiv` integer-literal
              | affine-expr `floordiv` integer-literal
              | affine-expr `mod` integer-literal
              | `-`affine-expr
              | bare-id
              | `-`? integer-literal

multi-dim-affine-expr ::= `(` `)`
                        | `(` affine-expr (`,` affine-expr)* `)`
````

我们区分 "字面量" 和 "符号"/"常量". 字面量是能在编译时就确定的常数, 符号/常量是只能在运行时确定, 但是在循环过程中不会改的量. 仿射变换的除法和取余(~~模法~~)只能对字面量而不是符号使用.

### 半仿射变换

半仿射变换 (semi-affine maps) 是要求放宽的仿射变换, 多加了对除法和取余使用符号的能力.

### 整数集合/条件

用来表示维度和符号应当满足的条件的:

````
integer-set-id ::= `#` suffix-id

integer-set-inline
   ::= dim-and-symbol-id-lists `:` '(' affine-constraint-conjunction? ')'

// Declarations of integer sets are at the top of the file.
integer-set-decl ::= integer-set-id `=` integer-set-inline

// Uses of integer sets may use the inline form or the named form.
integer-set ::= integer-set-id | integer-set-inline
````

例子:

````mlir
// A example two-dimensional integer set with two symbols.
#set42 = affine_set<(d0, d1)[s0, s1]
   : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s1 - 1 >= 0)>

// Inside a Region
affine.if #set42(%i, %j)[%M, %N] {
  ...
}
````

## 操作

TODO
