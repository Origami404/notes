The internal representation of an operation is simple: an operation is identified by a unique string (e.g. `dim`, `tf.Conv2d`, `x86.repmovsb`, `ppc.eieio`, etc), can return zero or more results, take zero or more operands, has a dictionary of [Attributes](Attributes.md), has zero or more successors, and zero or more enclosed [Region](Region.md)s.

一个操作有一个唯一的字符串充当标识符, 可以返回 0 或任意多个结果, 可以接受 0 或任意多个参数, 拥有[属性](Attributes.md) (一个 dict), 有 0 或任意多个后继, 0 或任意多个[区域](Region.md).

操作的默认文本形式输出如下:

The generic printing form includes all these elements literally, with a function type to indicate the types of the results and operands.

````
operation            ::= op-result-list? (generic-operation | custom-operation)
                         trailing-location?
                         
generic-operation    ::= string-literal `(` value-use-list? `)`  successor-list?
                         region-list? dictionary-attribute? `:` function-type
                         
custom-operation     ::= bare-id custom-operation-format

op-result-list       ::= op-result (`,` op-result)* `=`
op-result            ::= value-id (`:` integer-literal)

successor-list       ::= `[` successor (`,` successor)* `]`
successor            ::= caret-id (`:` block-arg-list)?

region-list          ::= `(` region (`,` region)* `)`
dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
trailing-location    ::= (`loc` `(` location `)`)?
````

例子:

````mlir
// "foo_div" 是标识符
// () 代表传给它 0 个参数
// : 后面是类型, () -> (f32, i32) 说明它接受 0 个参数, 返回两个返回值
// result 是存放返回值的, :2 代表它存着两个结果
// 如果后面想要拿到特定位置的结果, 可以用 %result#0, %result#1 的方式
%result:2 = "foo_div"() : () -> (f32, i32)

// 可以直接解构赋值
%foo, %bar = "foo_div"() : () -> (f32, i32)

// 带参数和属性的调用
%2 = "tf.scramble"(%result#0, %bar) {fruit = "banana"} : (f32, i32) -> f32
````
