
````
// Identifiers
bare-id ::= (letter|[_]) (letter|digit|[_$.])*
bare-id-list ::= bare-id (`,` bare-id)*
value-id ::= `%` suffix-id
alias-name :: = bare-id
suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
value-id-list ::= value-id (`,` value-id)*

// Uses of value, e.g. in an operand list to an operation.
value-use ::= value-id
value-use-list ::= value-use (`,` value-use)*
````

Identifiers name entities such as values, types and functions, and are chosen by the writer of MLIR code. Identifiers may be descriptive (e.g. `%batch_size`, `@matmul`), or may be non-descriptive when they are auto-generated (e.g. `%23`, `@func42`). Identifier names for values may be used in an MLIR text file but are not persisted as part of the IR - the printer will give them anonymous names like `%42`.

标识符可以是描述性的, 也可以是自动生成的递增数字. 它只是在文本形式里被使用, 而不会存在于 IR 的内存形式里.

MLIR guarantees identifiers never collide with keywords by prefixing identifiers with a sigil (e.g. `%`, `#`, `@`, `^`, `!`). In certain unambiguous contexts (e.g. affine expressions), identifiers are not prefixed, for brevity. New keywords may be added to future versions of MLIR without danger of collision with existing identifiers.

标识符一般会被一个前缀修饰, 它标识的实体的类型确定了这个前缀. 在特定无歧义的上下文之中, 这个前缀可以被省略.
