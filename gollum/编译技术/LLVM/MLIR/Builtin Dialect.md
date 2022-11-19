Ref: <https://mlir.llvm.org/docs/Dialects/Builtin/>

The builtin dialect contains a core set of Attributes, Operations, and Types that have wide applicability across a very large number of domains and abstractions. Many of the components of this dialect are also instrumental in the implementation of the core IR. As such, this dialect is implicitly loaded in every `MLIRContext`, and available directly to all users of MLIR.

它是 MLIR 的 "核心部分", 里面定义了基本上所有方言都有可能需要的设施, 默认对所有方言可用, 并且允许出现在任何结果中.

### 操作

* `builtin.module`: 接受一个区域, 充当 "module" 使用.
* `builtin.unrealized_conversion_cast`: 从任意类型的值到任意其它类型的未知实现的转换. 可以接受 0-N 个参数, 返回 1-N 个值.

````mlir
module {
  func.func @foo()
}

%result = unrealized_conversion_cast to !bar.tuple_type<>
%result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type
%results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type
%result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>
````

### 类型

非常, 非常多的基本类型

* 很基本的类型
  * `FloatNNNType`, 其中 `NNN` 是 16, 32, 64, 80, 128
  * `FunctionType`
  * `IndexType`
  * `IntegerType`: 从 `i1` 到 `i114514` 都算这个
  * `MemRefType`: 代表某块内存空间的引用的类型, 可以任意维度跟阶, 和张量差不多 (`memref<1 x 0 x f32>`)
    * `UnrankedMemRefType`, 长度不确定版本, `memref<*xf32>`
    * 多维的还可以带一个 layout, 指定诸如行优先还是列优先之类的, 但是更加地灵活, 可以视作 "索引到实际位置的映射"
  * `NoneType`: 是 `()`, 不是 `absurd`. 有且只有一个值.
  * `TupleType`
* 不算很基本的类型
  * `ComplexType`
  * `BFloat16Type`: 常用于人工智能领域的一种 16 bit 浮点数格式. 砍了 16 bit 的一点精度的位置使得其能表示的范围跟 32 bit 的差不多
  * `Float8E5M2Type`: 非常奇葩的一种浮点数格式, 又称为 `S1E5M2` 格式, 也就是 1 位符号位, 5 位指数位, 2 位小数位, 估计又是什么人工智能玩意
  * `OpaqueType`: 代表某种外部类型
  * `RankedTensorType`: 固定阶的任意维张量
  * `UnrankedTensorType`: 不固定阶的张量
  * `VectorType`: SIMD 用的向量类型
    * `vector<3x42xi32>`: 二维的 i32 向量, 可能一些 TPU 会收
    * `vector<[4]xf32>`: 长度为 4 的不定长的 f32 向量
    * `vector<[2x8]xf32>`: 以 2x8 为单位的不定长二维向量
    * `vector<4x[4]xf32>`: 4 个长度为 4 的不定长向量组成的二维向量

### 属性

提供了很多基本对象作为属性的包装. 比如说:

* 基本类型的包装属性:
  * `DictionaryAttr`
  * `FloatAttr`
  * `IntegerAttr`
  * `ArrayAttr`
  * `OpaqueAttr`: 其它的暂时未知方言所属的属性的文本形式
  * `StringAttr`
  * `TypeAttr`
  * `UnitAttr`: 单独的一个原子属性, 跟原子/关键字差不多?
* 仿射相关:
  * TODO
* 位置信息
  * `CallSiteLoc`: 关联一次函数调用的 caller 与 callee
  * `FileLineColLoc`
  * `FusedLoc`: 任何其它的跟源码位置有关, 但是又不属于已有类别的信息
  * `NameLoc`: 一个被命名的源码位置
  * `OpaqueLoc`: 保存一个外部的, 记录的位置信息的结构的指针, 对 MLIR 不透明
  * `UnknownLoc`: 表示未知的位置, 一般占位用

````mlir
// 基本对象
[10, i32]
{attr_name = "string attribute"}
42.0, 10
#dialect<"opaque attribute data">
"An important string"
i32, !dialect.type
func.func @simple_form() attributes {dialectName.unitAttr}

// 仿射相关
TODO

// 位置信息
loc(callsite("foo" at "mysource.cc":10:8))
loc("mysource.cc":10:8)
loc(fused["mysource.cc":10:8, "mysource.cc":22:8)
loc(fused<"CSE">["mysource.cc":10:8, "mysource.cc":22:8])
loc("CSE"("mysource.cc":10:8))
loc(?)
````
