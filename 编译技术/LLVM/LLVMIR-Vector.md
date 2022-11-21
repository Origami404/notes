LLVM 里的向量类型是一个第一类类型, 这意味着它并非任何的聚合类型, 不能使用 gep 等指令来获取内部成员, 只能通过有限的操作来对其作出修改.

此类型代表一个任意长度的, 可向量化的操作. 它逻辑上包含三个部分:

* size: 元素数量
* data type: 元素的类型
* (可变的) 表示底层硬件支持的向量化长度的属性 (这在 LLVM IR 层面一般不知道)

## 内存布局

一般而言, 内部元素类型如果是按直接字节对齐的, 那么就跟数组一样普普通通地存. 否则, 它会被 bitcast 成一个总比特数这么长的整数.

````llvm
%val = bitcast <4 x i4> <i4 1, i4 2, i4 3, i4 5> to i16

; Bitcasting from a vector to an integral type can be seen as
; concatenating the values:
;   %val now has the hexadecimal value 0x1235.

store i16 %val, ptr %ptr

; In memory the content will be (8-bit addressing):
;
;    [%ptr + 0]: 00010010  (0x12)
;    [%ptr + 1]: 00110101  (0x35)
````

向量的类型书写如下:

````
< <# elements> x <elementtype> >          ; Fixed-length vector
< vscale x <# elements> x <elementtype> > ; Scalable vector
````

例子:

````
<4 x i32>     ; 4 个 i32
<8 x float>   ; 8 个 float
<2 x i64>     ; 2 个 i64
<vscale x 4 x i32> ; n 个 4 个 i32, 相当于 4n 个 i32
````

## 基本操作

### `extractelement`

从向量中取出特定索引的元素

````
<result> = extractelement <n x <ty>> <val>, <ty2> <idx>          ; yields <ty>
<result> = extractelement <vscale x n x <ty>> <val>, <ty2> <idx> ; yields <ty>
````

### `insertelement`

获得一个给定位置是给定值的新向量. 它不会改变向量长度.

````
; yields <n x <ty>>
<result> = insertelement <n x <ty>> <val>, <ty> <elt>, <ty2> <idx>    

; yields <vscale x n x <ty>>
<result> = insertelement <vscale x n x <ty>> <val>, <ty> <elt>, <ty2> <idx> 
````

````llvm
<result> = insertelement <4 x i32> %vec, i32 1, i32 0    ; yields <4 x i32>
````

### `shufflevector`

从两个输入向量中, 取出特定位置的元素, 组合成一个新的向量. 这些位置必须是编译期已知的量. 对于 vscale 的向量而言, 只有 `zeroinitializer` 和 `undef` 可以作为 `mask`, 因为它的长度在编译期不可知.

````
; yields <m x <ty>>
<result> = shufflevector <n x <ty>> <v1>, <n x <ty>> <v2>, <m x i32> <mask>    

; yields <vscale x m x <ty>>
<result> = shufflevector <vscale x n x <ty>> <v1>, <vscale x n x <ty>> v2, <vscale x m x i32> <mask>  
````

````llvm
; yields <4 x i32>
; 若 %v1: [0, 1, 2, 3], %v2: [4, 5, 6, 7]
; 返回 [0, 4, 1, 5]
<result> = shufflevector <4 x i32> %v1, <4 x i32> %v2,
                        <4 x i32> <i32 0, i32 4, i32 1, i32 5> 

; yields <4 x i32> - Identity shuffle.
<result> = shufflevector <4 x i32> %v1, <4 x i32> undef,
                        <4 x i32> <i32 0, i32 1, i32 2, i32 3>  

; yields <4 x i32>
<result> = shufflevector <8 x i32> %v1, <8 x i32> undef,
                        <4 x i32> <i32 0, i32 1, i32 2, i32 3>  

; yields <8 x i32>
<result> = shufflevector <4 x i32> %v1, <4 x i32> %v2,
                        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >  
````

## 向量化相关操作

通过各种 intrinsics 来实现向量化.

* [Vector Reduction Intrinsics](https://llvm.org/docs/LangRef.html#vector-reduction-intrinsics)
* [Vector Predication Intrinsics](https://llvm.org/docs/LangRef.html#id2027)
