Each value in MLIR has a type defined by the type system.

MLIR 里的每一个值都有其类型. 类型可以是 别名/方言定义的类型/内建类型

````
type ::= type-alias | dialect-type | builtin-type
````

### 用法

基本上就是说, 每一个操作后面可以用 `:` 跟一个类型. 函数的参数和返回类型都可以是一个类型列表或是单个类型.

````
type-list-no-parens ::=  type (`,` type)*
type-list-parens ::= `(` `)`
                   | `(` type-list-no-parens `)`

// This is a common way to refer to a value with a specified type.
ssa-use-and-type ::= ssa-use `:` type
ssa-use ::= value-use

// Non-empty list of names and types.
ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*

function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
````

### 类型别名

以 `!` 开头的 "赋值" 语句就是类型别名. 使用别名的时候也要加感叹号.

````
type-alias-def ::= '!' alias-name '=' type
type-alias ::= '!' alias-name
````

例子:

````mlir
!avx_m128 = vector<4 x f32>

// Using the original type.
"foo"(%x) : vector<4 x f32> -> ()

// Using the type alias.
"foo"(%x) : !avx_m128 -> ()
````

### 方言定义的类型

````
dialect-namespace ::= bare-id

// 必须是感叹号开头, 可以是普通版或是好看版
dialect-type ::= '!' (opaque-dialect-type | pretty-dialect-type)
// 普通版就是简单的 "方言名<内容>"; 好看版可以是 "方言名.xxx.yyy.zzz<内容>"
opaque-dialect-type ::= dialect-namespace dialect-type-body
pretty-dialect-type ::= dialect-namespace '.' pretty-dialect-type-lead-ident
                                              dialect-type-body?
pretty-dialect-type-lead-ident ::= '[A-Za-z][A-Za-z0-9._]*'

// 内容里还可以嵌套 <>, (), [], {}, 并且使用任意的东西
dialect-type-body ::= '<' dialect-type-contents+ '>'
dialect-type-contents ::= dialect-type-body
                            | '(' dialect-type-contents+ ')'
                            | '[' dialect-type-contents+ ']'
                            | '{' dialect-type-contents+ '}'
                            | '[^\[<({\]>)}\0]+'
````

````mlir
// A tensorflow string type.
!tf<string>

// A type with complex components.
!foo<something<abcd>>

// An even more complex type.
!foo<"a123^^^" + bar>
````

下面的好看版使用方式跟上面的等价:

````mlir
// A tensorflow string type.
!tf.string

// A type with complex components.
!foo.something<abcd>
````

### 内建类型

就是[内建方言](Builtin%20Dialect.md#lei-xing)里定义的类型.
