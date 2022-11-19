Attributes are the mechanism for specifying constant data on operations in places where a variable is never allowed - e.g. the comparison predicate of a [`cmpi` operation](https://mlir.llvm.org/docs/Dialects/ArithOps/). Each operation has an attribute dictionary, which associates a set of attribute names to attribute values. MLIR’s builtin dialect provides a rich set of [builtin attribute values](https://mlir.llvm.org/docs/LangRef/#builtin-attribute-values) out of the box (such as arrays, dictionaries, strings, etc.). Additionally, dialects can define their own [dialect attribute values](https://mlir.llvm.org/docs/LangRef/#dialect-attribute-values).

属性是 MLIR 里用于为操作指定编译期可知信息的机制. 属性里不允许使用变量. 属性的一个常见用法是在 `cmpi` (整数比较指令) 里指定比较方式 (如 `gt`, `lt`, `eq` 等).

每一个操作都可以拥有它自己的属性, 表示为一个键值对字典的形式. 我们区分属性本身与属性字典 -- 后者是附着在每一个操作上的, 存储属性-值映射的字典.

属性可以是 属性别名/方言属性/内建属性, 与类型的定义非常类似.

````
// 属性字典里的条目
attribute-entry ::= (bare-id | string-literal) `=` attribute-value
// 属性的种类
attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
````

The top-level attribute dictionary attached to an operation has special semantics. The attribute entries are considered to be of two different kinds based on whether their dictionary key has a dialect prefix:

* *inherent attributes* are inherent to the definition of an operation’s semantics. The operation itself is expected to verify the consistency of these attributes. An example is the `predicate` attribute of the `arith.cmpi` op. These attributes must have names that do not start with a dialect prefix.
* *discardable attributes* have semantics defined externally to the operation itself, but must be compatible with the operations’s semantics. These attributes must have names that start with a dialect prefix. The dialect indicated by the dialect prefix is expected to verify these attributes. An example is the `gpu.container_module` attribute.

属性有两种, *固有属性* 与 *外界属性*. 区别如下:

* 固有属性代表了操作本身不可或缺的一部分信息, 改变它直接改变整个操作的语义. 比如说 前面提到的 `cmpi` 的 `lt` 等属性就是固有属性. 在属性字典里的固有属性不会带方言的命名空间前缀. (因为它本身就是跟操作强绑定的, 可以认为它直接属于这个操作, 不需要再进一步指定了)
* 外界属性并不是操作本身语义的一部分, 而是代表了对操作的 "修正" 或 "混入 (mixin)". 在属性字典里使用外界属性的时候必须带上属性定义时所属的方言的命名空间前缀. 对这种属性的兼容性/语义检查由定义它的方言来做.

### 属性别名

用 `#` 开头的 "赋值" 语句定义了新的属性别名. 

````
attribute-alias-def ::= '#' alias-name '=' attribute-value
attribute-alias ::= '#' alias-name
````

下面是一个令我迷惑的例子, 虽然能看懂它的意思就是了.

````mlir
#map = affine_map<(d0) -> (d0 + 10)>

// 这个原来的用法是什么鬼?
// Using the original attribute.
%b = affine.apply affine_map<(d0) -> (d0 + 10)> (%a)

// Using the attribute alias.
%b = affine.apply #map(%a)
````

### 方言定义的属性

基本上跟[方言定义的类型](Type.md#fang-yan-ding-yi-de-lei-xing)一样, 有普通版跟好看版; 甚至尖括号里能放的内容都是一样的.

````
dialect-namespace ::= bare-id

dialect-attribute ::= '#' (opaque-dialect-attribute | pretty-dialect-attribute)
opaque-dialect-attribute ::= dialect-namespace dialect-attribute-body
pretty-dialect-attribute ::= dialect-namespace '.' pretty-dialect-attribute-lead-ident
                                              dialect-attribute-body?
pretty-dialect-attribute-lead-ident ::= '[A-Za-z][A-Za-z0-9._]*'

dialect-attribute-body ::= '<' dialect-attribute-contents+ '>'
dialect-attribute-contents ::= dialect-attribute-body
                            | '(' dialect-attribute-contents+ ')'
                            | '[' dialect-attribute-contents+ ']'
                            | '{' dialect-attribute-contents+ '}'
                            | '[^\[<({\]>)}\0]+'
````

至于普通版记法跟好看版记法的转换, 跟[方言定义的类型](Type.md#fang-yan-ding-yi-de-lei-xing)也是一样的.

### 内建属性

就是[内建方言的属性](Builtin%20Dialect.md#shu-xing).
