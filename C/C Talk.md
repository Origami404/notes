## 读懂声明

* 如何使用就如何声明
* 各种复合类型
* typedef

## 声明的意义

* 声明与定义
* 信息 == 位 + 上下文

## C 语言基础设施

## Modern C 的流程

* 总览: 声明, 定义, 对象, 标识符
* 控制流
* 表达式
  * size_t 是自然数, 而不是 int
  * 溢出
  * For unsigned values, `a == (a/b)*b + (a%b)`

## C 标准

* value: precise meaning of the contents of an object when interpreted as having a specific type
  * trap representation: an object representation that need not represent a value of the object type
* object: region of data storage in the execution environment, the contents of which can represent values
  * When referenced, an object may be interpreted as having a particular type; see 6.3.2.1.
