The `memref` dialect is intended to hold core memref creation and manipulation ops, which are not strongly associated with any particular other dialect or domain abstraction.

MemRef 类型在 [Builtin Dialect](Builtin%20Dialect.md) 中被定义.

## 分配/读取/写入

### `memref.alloc`

````
`memref.alloc` `(`$dynamicSizes`)` (`[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
````

分配内存. 对于定长的 memref, 可以直接留空 `$dynamicSizes`, 直接靠类型信息分配; 对于不定长的就要给这个. 后边的 `$symbolOperands` 部分是给 affine 那边用的, 可以不用管.

````mlir
// 分配内存, 直接通过后面的 memref 类型推断
%0 = memref.alloc() : memref<8x64xf32, 1>
// 分配内存, %d 被绑定到后面类型中的 ?
%1 = memref.alloc(%d) : memref<8x?xf32, 1>
// 带对齐地分配内存
%2 = memref.alloc() {alignment = 8} : memref<8x64xf32, 1>
````

### `memref.alloca`

````
`memref.alloca` `(`$dynamicSizes`)` ([` $symbolOperands^ `]`)? attr-dict  `:` type($memref)
````

在栈上分配内存. 使用同 \[\[MemoryRef Dialect#`memref.alloc`\|`memref.alloc`\]\].

### `memref.alloca_scope`

带作用域的栈上内存分配. 常见语言 (如 C++) 中, 经常在函数体内单开一个代码块来声明变量, 使得析构/资源释放等在离开该代码块后执行. 该操作的意图与其类似.

 > 
 > 无语法文档, 自己补充的

````
memref.alloca_scope $bodyRegion
````

````mlir
memref.alloca_scope {
  // 控制流每次流到这里就会进行一次内存分配
  %myalloca = memref.alloca(): memref<4x3xf32>
  ...
  // 流出了就没有了
}
````

````mlir
// %result 就是 %bodyRegion 里通过 memref.alloca_scope.return 返回的值 (即 %value)
%result = memref.alloca_scope {
  ...
  // 可以返回一个值
  memref.alloca_scope.return %value
  // 如果没有返回值的话, 可以直接不写 memref.alloca_scope.return 
}
````

### `memref.realloc`

````
`memref.realloc` $source (`(` $dynamicResultSize^ `)`)? attr-dict
              `:` type($source) `to` type(results)
````

调整 MemRef 的大小. 用法与注意事项如下:

````mlir
// 基本用法
%0 = memref.realloc %src : memref<64xf32> to memref<124xf32>
// 将未知大小的内存调整成已知的, 多丢少补
%1 = memref.realloc %src : memref<?xf32> to memref<124xf32>
// 完全动态的调整, 目标大小由 %d 决定
%2 = memref.realloc %src(%d) : memref<?xf32> to memref<?xf32>
// 带对齐的 realloc
%3 = memref.ralloc %src {alignment = 8} : memref<64xf32> to memref<124xf32>

// realloc 之后的旧内存不应该再使用
// 因为在大部分实现上, realloc 扩大内存会导致重新复制
%new = memref.realloc %old : memref<64xf32> to memref<124xf32>
%4 = memref.load %new[%index]   // ok
%5 = memref.load %old[%index]   // undefined behavior
````

### `memref.load`

````
`memref.load` $memref `[` $indices `]` attr-dict `:` type($memref)
````

获得 `$memref[$indices]` 中的数据.

### `memref.dealloc`

````
`memref.dealloc` $memref attr-dict `:` type($memref)
````

回收内存. 注意它显然不能对内存视图 (即没有所有权的内存) 做释放, 这显然会导致 double-free.

## 转换

### `memref.cast`

````
`memref.cast` $source attr-dict `:` type($source) `to` type($dest)
````

 > 
 > 为什么官方文档有两个 syntax?

实际上不能说是强转吧, 它只能在相容类型之间换. 而相容的要求非常严格:

* 必须有同样的基类型
* 具有相容的布局
  * 相同的布局是相容的 (即 4x4 的只能转到 4x4 的, 不能转到 16 的)
  * 带 `?` 的不确定的可以转过去也可以转过来 (比如 ?x4 可以转到 4x4, 同样不能转到 16)
  * 带不定阶的可以转过去也可以转过来 (比如 * 可以转到 4x4 或 16 或 2x2x2x?)

````mlir
// 从 ? 转到确定数值的过程相当于断言; 如果失败会丢运行时错误
%2 = memref.cast %1 : memref<?x?xf32> to memref<4x4xf32>
// Erase static shape information, replacing it with dynamic information.
%3 = memref.cast %1 : memref<4xf32> to memref<?xf32>

// The same holds true for offsets and strides.

// 不定阶 (unranked) 的转换
%4 = memref.cast %1 : memref<*xf32> to memref<4x?xf32>
%5 = memref.cast %1 : memref<4x?xf32> to memref<*xf32>

// 如果真的需要从 4x4 转到 16 的话, 需要这样: 4x4 -> * -> 16
// 并且也不能动基元素类型, 不可用把 i32 转成 f32
````

### `memref.reinterpret_cast`

能设定所有元信息 (阶, 各个维度, 各个偏移量, 各个步长) 的转换. 还是不能转换基类型.

````
`memref.reinterpret_cast` $source `to` `offset` 
            `:` custom<DynamicIndexList>($offsets, $static_offsets, "ShapedType::kDynamicStrideOrOffset") `,` `sizes` 
            `:` custom<DynamicIndexList>($sizes, $static_sizes, "ShapedType::kDynamicSize") `,` `strides` 
            `:` custom<DynamicIndexList>($strides, $static_strides, "ShapedType::kDynamicStrideOrOffset")
              attr-dict `:` type($source) `to` type($result)
````

 > 
 > custom\<\> 表示一个完全定制化的文法与数据类型

````mlir
memref.reinterpret_cast %ranked to
  offset: [0],
  sizes: [%size0, 10],
  strides: [1, %stride1]
: memref<?x?xf32> to memref<?x10xf32, strided<[1, ?], offset: 0>>

memref.reinterpret_cast %unranked to
  offset: [%offset],
  sizes: [%size0, %size1],
  strides: [%stride0, %stride1]
: memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
````

### `memref.reshape`

改变一个 MemRef 的形状 (即阶数与各阶维度长度). 下面提到的 \[\[MemoryRef Dialect#`memref.collapse_shape`\|`memref.collapse_shape`\]\] 和 \[\[MemoryRef Dialect#`memref.expand_shape`\|`memref.expand_shape`\]\] 同样是改变形状, 但是它们只能用一个字面量来指定如何合并或拆分 MemRef 的

### `memref.collapse_shape`

在 C 语言里, 我们经常把一个二维数组 `a[N][M]` 压扁成一个一维数组 `a[N*M]`, 这种操作可以视为维度的坍缩 (collapse). 这个操作也是同理, 它允许你将一段连续的索引坍缩成一个.

**此操作不会创建新内存! 返回的是对旧内存的一个视图! 不要对它 dealloc!**

````
`memref.collapse_shape` $src $reassociation attr-dict `:` type($src) `into` type($result)
````

其中 `$reassociation` 指定了坍缩的方式, 例子如下:

````mlir
// Dimension collapse (i, j) -> i' and k -> k'
// [[0, 1], [2]]: [0, 1] 代表结果第 0 个维度是原来的第 0 个和第 1 个维度坍缩成的; [2] 代表结果的第 1 个维度是原来的第 2 个维度坍缩成的 
%1 = memref.collapse_shape %0 [[0, 1], [2]] :
    memref<?x?x?xf32, stride_spec> into memref<?x?xf32, stride_spec_2>
````

只能对后边跟着的, 连续的维度进行坍缩, 因为这种缩才是自然的缩. 显然我们不可能在不复制内存的情况下, 将一个三维数组 `a[A][B][C]` 缩成 `a[A * C][B]` 之类的, 只能缩成 `a[A*B][C]`, `a[A][B*C]`, `a[A*B*C]`.

特别地, 对于一个任意维度都只有单位长度的 MemRef, 它还可以坍缩成一个 0 阶的 MemRef, 此时 `$reassociation` 为空 (`[]`):

````mlir
%2 = memref.collapse_shape %unit [] : memref<1x1x1xf32> into memref<f32>
````

### `memref.expand_shape`

升维操作, 是 \[\[MemoryRef Dialect#`memref.collapse_shape`\|`memref.collapse_shape`\]\] 的逆操作. 同样也只是旧内存的一个 **视图** 而非新内存. 同样地也只能将一个维度拆分成连续的多个维度.

````
`memref.expand_shape` $src $reassociation attr-dict `:` type($src) `into` type($result)
````

````mlir
// 原值的第 0 个维度被拆分新值的第 0, 1 个维度
// 原值的第 1 个维度成为新值的第 2 个维度
// 对于新拆分出来的维度组, 最多只能有一个维度是 ? 的, 否则拆分就有歧义了
%r = memref.expand_shape %0 [[0, 1], [2]]
    : memref<?x?xf32> into memref<?x5x?xf32>
````

如果拆不开 (比如尝试把 `10` 拆成 `2x6`), 会有编译错或者运行错. 

因为对 `?` 维度长度的断言已经有 \[\[MemoryRef Dialect#`memref.cast`\|`memref.cast`\]\] 在做了, 所以它只能把带 `?` 的维度转换成至少带一个 `?` 的维度组. 也就是说, `?x4` 不能直接用本指令转成 `16x2`, 只能用本指令先转成 `?x2`, 然后用再 cast 转成 `16x2`.

这个操作同样也有对 0 阶 MemRef 的使用 `[]` 的特例:

````mlir
%unit = memref.expand_shape %2 [] : memref<f32> into memref<1x1x1xf32>
````

## 普通操作

### `memref.rank`

获得 MemRef 的阶数.

````
`memref.rank` $memref attr-dict `:` type($memref)
````

对于 Unranked MemRef (带 `*` 的), 其不是常量, 需要运行时确定. 对于其他的正常 MemRef 而言, 这个应该都是一个常量.

### `memref.dim`

````
`memref.dim` attr-dict $source `,` $index `:` type($source)
````

获取 `$source` 对应维度 `$index` 的大小. 可以获取静态的也可以获取动态的. 如果 `$index` 超过了阶数, 那么是未定义行为.

````mlir
// 可以用来获取常量大小, 此时 %x 是一个常量, 可以参与常量折叠
%c0 = arith.constant 0 : index
%x = memref.dim %A, %c0 : memref<4 x ? x f32>

// 也可以用来获取 ? 维度的长度
%c1 = arith.constant 1 : index
%y = memref.dim %A, %c1 : memref<4 x ? x f32>
````

### `memref.copy`

````
`memref.copy` $source `,` $target attr-dict `:` type($source) `to` type($target)
````

复制数据.

### `memref.global`

````
`memref.global` ($sym_visibility^)? (`constant` $constant^)? $sym_name 
    `:` custom<GlobalMemrefOpTypeAndInitialValue>($type, $initial_value)
    attr-dict
````

 > 
 > custom\<\> 表示一个完全定制化的文法与数据类型

声明或定义一个全局的 MemRef 变量. 严格来讲, 是一个可以拿到 MemRef 的全局变量, 它本身似乎并不能直接当全局变量使用. 如果没有 `$initial_value`, 这就是一个声明; 有那就是一个定义. 如果需要一个定义, 但是又不想初始化一个全局变量, 那可以使用 `uninitialized`.

````mlir
// 私有全局变量定义
memref.global "private" @x : memref<2xf32> = dense<0.0,2.0>

// 带对齐的私有全局变量定义
memref.global "private" @x : memref<2xf32> = dense<0.0,2.0> {alignment = 64}

// 私有的全局变量声明, 可以在其他地方定义
memref.global "private" @y : memref<4xi32>

// 全局变量定义, 但是是未初始化的
memref.global @z : memref<3xf16> = uninitialized

// 外部可见的常量全局变量
memref.global constant @c : memref<2xi32> = dense<1, 4>
````

### `memref.get_global`

````
`memref.get_global` $name `:` type($result) attr-dict
````

从全局变量中获得一个 MemRef.

 > 
 > 为什么全局变量要单走一个呢? MLIR 又不需要翻译 C 类语言这种全局变量天然可变的存在, 为什么不直接让全局变量成为全局 Region 的变量呢?

## Lowering 相关

### `memref.extract_aligned_pointer_as_index`

````
`memref.extract_aligned_pointer_as_index` $source 
    `:` type($source) `->` type(results) attr-dict
````

获得指向内存的指针, 以 index 类型的形式. 一般就用在 lowering 相关操作里, 这样就不用为每一个其它方言都加入一个 `memref.to_xxx_ptr` 或者 `xxx.from_memref_ptr` 了. 大家都转成 `i64` 然后互丢就完事了.

````
// lowering 到 LLVM Dialect
%0 = memref.extract_aligned_pointer_as_index %arg : memref<4x4xf32> -> index
%1 = arith.index_cast %0 : index to i64
%2 = llvm.inttoptr %1 : i64 to !llvm.ptr<f32>
call @foo(%2) : (!llvm.ptr<f32>) ->()
````

### `memref.extract_strided_metadata`

````
`memref.extract_strided_metadata` $source `:` type($source) `->` type(results) attr-dict
````

它可以一口气获取一个 MemRef 全部的元信息:

````mlir
// 基指针, 偏移量, 各维度大小, 各维度步幅
%base, %offset, %sizes:2, %strides:2 =
memref.extract_strided_metadata %memref :
  memref<10x?xf32>, index, index, index, index, index

// After folding, the type of %m2 can be memref<10x?xf32> and further
// folded to %memref.
%m2 = memref.reinterpret_cast %base to
  offset: [%offset],
  sizes: [%sizes#0, %sizes#1],
  strides: [%strides#0, %strides#1]
: memref<f32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
````

## 优化相关

### `memref.assume_alignment`

````
`memref.assume_alignment` $memref `,` $alignment attr-dict `:` type($memref)
````

其仅用于标记 `$memref` 内部是对齐到 `$alignment` 的, 没有任何操作. 如果实际上 `$memref` 不是对齐的, 那么是未定义行为.

 > 
 > 为什么不用属性直接在当初的 `$memref` 上标记呢?

### `memref.prefetch`

指示接下来我们要访问/读取哪个 MemRef 的哪个位置, 并且标识一下它的局域性有多高.

 > 
 > 没有文法, 自己编的

````
`memref.prefetch` $memref `[` $indices `]` `,` $specifier `,` $locality `,` $cacheType `:` type($memref)
````

例子:

````mlir
memref.prefetch %0[%i, %j], read, locality<3>, data
    : memref<400x400xi32>
````

* `$specifier` 只能是 `read` 或 `write`, 标识接下来是要读取这个位置还是写入这个位置.
* `$locality` 局域性的等级, 可以是从 `locality<0>` 到 `locality<3>` 的四个值, `3` 代表需要最高的局域性, `0` 代表不需要.
* `$cacheType` 只能是 `data` 或 `instr`, 标志此缓存应该是被当作数据缓存还是指令缓存.

## 并发相关

### `memref.generic_atomic_rmw`

 > 
 > 无语法文档, 自己补充的

````
`memref.generic_atomic_rmw` $memref `[` $indices `]` : type($memref) $atomic_body
````

原子地对 `$memref[$indices]` 进行一次操作, 操作即为区域 `$atomic_body` 里面的指令, 按控制流执行. 对应的内存区域的原值被当作参数传给 `$region`, 随后接受从 `$atomic_body` 里通过 \[\[MemoryRef Dialect#`memref.atomic_yield`\|`memref.atomic_yield`\]\] 传出的值, 用之更新 `$memref[$indices]`. 结果是最后一个向 `$memref[$indices]` 里写入的值.

````mlir
%x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
  ^bb0(%current_value : f32):
    %c1 = arith.constant 1.0 : f32
    %inc = arith.addf %c1, %current_value : f32
    memref.atomic_yield %inc : f32
}
````

### `memref.atomic_yield`

````
`memref.atomic_yield` $result attr-dict `:` type($result)
````

具体用法见 \[\[MemoryRef Dialect#`memref.generic_atomic_rmw`\|`memref.generic_atomic_rmw`\]\].

### `memref.atomic_rmw`

````
`memref.atomic_rmw` $kind $value `,` $memref `[` $indices `]` attr-dict 
    `:` `(` type($value) `,` type($memref) `)` `->` type($result)
````

原子地对进行一次 "读-改-写" (Read-Modify-Write) 操作. 相当于执行 `$kind $value $memref[$indices]`. 可以认为是 \[\[MemoryRef Dialect#`memref.generic_atomic_rmw`\|`memref.generic_atomic_rmw`\]\] 的单条操作版.

下面是完成了 `memref.generic_atomic_rmw` 里的例子的等价操作:

````mlir
%x = memref.atomic_rmw "addf" %value, %I[%i] : (f32, memref<10xf32>) -> f32
````

`$kind` 的种类如下:

````python
# llvm/mlir/include/mlir/Dialect/Arith/IR/ArithBase.td:72

def ATOMIC_RMW_KIND_ADDF    : I64EnumAttrCase<"addf", 0>;
def ATOMIC_RMW_KIND_ADDI    : I64EnumAttrCase<"addi", 1>;
def ATOMIC_RMW_KIND_ASSIGN  : I64EnumAttrCase<"assign", 2>;
def ATOMIC_RMW_KIND_MAXF    : I64EnumAttrCase<"maxf", 3>;
def ATOMIC_RMW_KIND_MAXS    : I64EnumAttrCase<"maxs", 4>;
def ATOMIC_RMW_KIND_MAXU    : I64EnumAttrCase<"maxu", 5>;
def ATOMIC_RMW_KIND_MINF    : I64EnumAttrCase<"minf", 6>;
def ATOMIC_RMW_KIND_MINS    : I64EnumAttrCase<"mins", 7>;
def ATOMIC_RMW_KIND_MINU    : I64EnumAttrCase<"minu", 8>;
def ATOMIC_RMW_KIND_MULF    : I64EnumAttrCase<"mulf", 9>;
def ATOMIC_RMW_KIND_MULI    : I64EnumAttrCase<"muli", 10>;
def ATOMIC_RMW_KIND_ORI     : I64EnumAttrCase<"ori", 11>;
def ATOMIC_RMW_KIND_ANDI    : I64EnumAttrCase<"andi", 12>;

def AtomicRMWKindAttr : I64EnumAttr<
    "AtomicRMWKind", "",
    [ATOMIC_RMW_KIND_ADDF, ATOMIC_RMW_KIND_ADDI, ATOMIC_RMW_KIND_ASSIGN,
     ATOMIC_RMW_KIND_MAXF, ATOMIC_RMW_KIND_MAXS, ATOMIC_RMW_KIND_MAXU,
     ATOMIC_RMW_KIND_MINF, ATOMIC_RMW_KIND_MINS, ATOMIC_RMW_KIND_MINU,
     ATOMIC_RMW_KIND_MULF, ATOMIC_RMW_KIND_MULI, ATOMIC_RMW_KIND_ORI,
     ATOMIC_RMW_KIND_ANDI]> {
  let cppNamespace = "::mlir::arith";
}
````

 > 
 > assign 是按位的还是按值的?\```
 > pip install obsidianhtml

````


## DMA 相关

DMA, Direct Memory Access, 是指绕开 CPU, 直接在内存内部将数组从一个地方搬到另一个地方的行为.

MemRef 中对 DMA 的支持由一系列 `dma_` 开头的操作支持. 

此部分 TODO.````
