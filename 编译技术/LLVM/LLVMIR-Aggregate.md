聚合类型. 包括数组和结构体.

## 类型

### 数组

语法:

````
[<# elements> x <elementtype>]
````

例子:

````
[40 x i32]
[2 x [3 x [4 x i16]]]
````

### 结构体

从内存布局上, 结构体分成打包用 (packed) 结构体和普通结构体. 普通结构体有对齐什么的, 而打包用的就无 padding, 默认全部按一字节对齐.

从用法上, 结构体分为具名 (identified) 结构体和字面量 (literal) 结构体.

````
%T1 = type { <type list> }     ; Identified normal struct type
%T2 = type <{ <type list> }>   ; Identified packed struct type
````

## 方法

### `extractvalue`

从聚合类型里获得特定成员. 它与 gep 的区别是:

* 索引直接是那个结构体, 而不是指针, 所以一开始的 0 可以省略
* 至少要有一个索引
* 结构体跟数组的索引都要在界内

例子:

````
; 从一个包含 i32 和 float 作为成员的结构体类型 {i32, float} 中提取第 0 个成员
<result> = extractvalue {i32, float} %agg, 0    ; yields i32
````

### `insertvalue`

与 \[\[LLVMIR-Vector#`insertelement`\]\] 类似, 它生成一个新聚合类型, 其中指定位置的值是新给的, 其它的跟旧的一样.

例子:

````
%agg1 = insertvalue {i32, float} undef, i32 1, 0              ; yields {i32 1, float undef}
%agg2 = insertvalue {i32, float} %agg1, float %val, 1         ; yields {i32 1, float %val}
%agg3 = insertvalue {i32, {float}} undef, float %val, 1, 0    ; yields {i32 undef, {float %val}}
````
