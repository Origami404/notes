From: <https://clang.llvm.org/docs/InternalsManual.html>

## LLVM Support Library

阅读[该文档](https://llvm.org/docs/ProgrammersManual.html)以获得 `libSupport` 的相关信息. 它负责:

* 命令行参数解析
* 各类 LLVM 独有的容器
* 跨平台抽象层 (在 clang 里主要被用于文件读写)

## The Clang "Basic" Library

### The Diagnostic Subsystem / 诊断信息

该部分负责报错和警告. 在 Clang 里, 一条诊断信息 (diagnostic) 大概由下面的部分构成:

* ID
* 一条英文的格式串 (Format String)
* 至少一个, 可以多个的 [SourceLocation](https://clang.llvm.org/docs/InternalsManual.html#sourcelocation) 指定该信息涉及到的源码位置
* 严重程度 (severity)
* 零个或多个参数, 可以以 `%0`, `%1`, ... 的形式嵌入到说明中

诊断信息只负责提供信息, 输出和格式由一个实现了 `DiagnosticConsumer` 接口的具体类来负责.

#### Severity

一条诊断信息的严重程度可以是:

* `NOTE`: 用来为之前的诊断提供更多信息的信息
* `REMARK`: 编译时产生的, 不一定与任何特定代码有关的信息
* `WARNING`: 有疑虑的代码
* `EXTENSION`: 不需要警告的 clang 扩展用法
* `EXTWARN`: 需要警告的 clang 扩展用法
* `ERROR`: 错误

这些严重程度在编译时视用户的编译选项和设置而定, 被划分为不同的展示等级 (output level): `Ignored`, `Note`, `Remark`, `Warning`, `Error`, `Fatal` (来自 `Diagnostic::Level`). 然后根据不同的展示等级被不同地对待/输出. 

Clang 提供了非常精细的命令行选项, 基本上爱怎么调整报错就怎么调整. 下面是两个约束:

* 严重程度为 `NOTE` 的信息的展示等级只能随它附加到的那条信息的展示等级而定
* 严重程度为 `ERROR` 的信息的展示等级必须为 `Fatal`

#### Format String
