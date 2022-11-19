### What is `stdc-predef.h`

It's a header file that will be implicit included by GCC. It contains 3 macro definition for feature test macros.

### 过程

14:00 - 15:00

* 创建了 reviews.llvm.org 的帐号, 通读了两个 Differential
* 拉取了新代码, 跑了两遍编译 (Debug+Release)
* 跑编译过程中阅读了 Contribution 与 Test 的文档, 并肉眼读了一下 patch
* 进行了测试, 根据测试报错安装了一些测试需要的 perl 包, 跑通了没加 patch 状态下的所有测试

15:00 - 16:00

* apply patch, 开始进行测试
* 测试在 Release 版本下通过了不应该通过的单元, 但是多失败了40个不应该失败的
* 测试在 Debug 版本下给出了期望行为

## 总体流程

* 找到 bug, 确认修复可能与基本方式
  * 日用
  * 翻之前的 issue
  * 看看现在的一些讨论激烈的 issue
  * review.llvm.org 注册帐号
* 进行编码
  * 多参考周围的
  * 多参考之前的
  * 找到 xxx 信息
* 运行测试
  * 编译与测试 
  * llvm-lit 与测试文件
  * 记得全量测试
* (可能的) 调试
* 编写详细的 commit-message
  * 总结
  * 做了什么, 没做什么
  * Link 与 Sign off by
* 发布
  * 格式化
  * arc 的使用
  * 选择合适的 reviewer

## 正文

### 发现问题

 > 
 > 代码不缺少改进的机会, 只缺少一双挑剔的眼睛. --by 马克 吐温

 > 
 > 当你想引用一句名言, 又不知道是谁说的, 就说是我说的. -- by 马克 吐温

#### 日用 Clang

如果您非常地有空, 并且有强劲的机器, 可以尝试安装 Gentoo, 然后将 clang 设置为默认的编译器. 这样您的日常使用便相当于对 Clang 进行的不间断的测试. 

#### 社区吃瓜

首先, 您可以在 [GitHub issue](), [LLVM 论坛](), [Phabricator]() 等地方寻找讨论比较激烈的串, 看看它们有没有给出一些之前被 revert 的 patch 的链接. 如果一个串内社区达成了共识, 并且有比较明确的指向之前存在过的 patch 的链接, 那么它也许就是一个很好的入手机会.

LLVM 的论坛 (`a`) 与 LLVM 用于 review patch 的地方 (Phabricator) 不是同一个地方. 

##### 注册 Phabricator 帐号

打开网站, 看着注册一下就可以了. 需要注意的是真名 (real name) 那一栏真的要写真名. 比如说您叫张三五, 那就得写 "SanWu Zhang".

如果您要交 patch 的话, 一个 Phabricator 帐号也是必须的.

#### 找一位 Mentor...

这个对运气和人脉的要求比较高. 您也许可以试着在您的大学或各种技术交流群内询问, 也许能找到曾经向 LLVM 提交过代码的大佬. 

 > 
 > 我的运气就非常好喵

### 进行编码

首先, 你可能需要我的 [Getting Started with Clang & LLVM](Getting%20Started%20with%20Clang%20&%20LLVM.md), 在本地准备好 LLVM 项目的源码, 然后, 就可以开始面对着一堆代码发呆了...

#### 参考之前的 Patch

如果你是准备重新 land 一份被 revert 的旧 patch, 那么首先应该参照的是旧的那个 patch. 

#### 参考旁边的代码

 > 
 > 当您突然混入了一堆陌生人身边, 想要让自己看起来比较正常, 最好的做法是什么呢? 那就是复读周围的人的话!

在大项目中编写代码时, 最先需要参考的便是 "附近的代码". 一般而言, 一个 Patch 很少会加入全新的功能, 更多地是参照之前的功能加入类似的, 或者是改变功能的行为. 这时候与自己实现功能类似的代码便是极好的参考材料.

#### 获得自己想要的信息

写代码时, 最常见的问题就是: "我该如何获取 xxx 信息?". 这种问题首先应该在代码中快速浏览一遍:

* 函数内的局部变量里有没有
* 如果是类方法, 那么类 (`this`) 里有没有
* 附近的代码有没有获取过类似的信息

如果浏览不到, 那么可以去:

* Clang Internal / LLVM Internal 中看看有没有对应子系统的总体介绍, 记住一些专有名词
* 在代码中搜索你觉得可能有这些信息的类或名字
* 在 Doxygen 中搜索

后两点可以范围由小到大交错进行, 比如先在当前文件搜一下, 不行就去 Doxygen 对应的类里搜, 再不行再去当前目录下搜索... 总之, 应该都是能搜到的.

一般而言, 在 Clang 中, 除开不可能保存的信息之外 (比如如果你尝试在 AST 阶段获得 Token 的信息, 那多半不行), 叠两到三层 getter 都应该能找到.

#### 快速在代码中进行跳转

首先, 至少要配好 clangd, 能找定义/引用. 其次, 最好熟练 vscode 的搜索和正则表达式搜索, 以及 Ctrl+P. 然后就是一个 [PinFile]() 插件, 它可以让你 "pin" 住你想关注的文件, 很适合交 patch 这种只需要关注一个庞大代码库的局部小部分的情况.

### 运行测试

首先, 在测试之前, 不要忘了重新编译!

建议先编译一个 Release 版本来运行测试, 这样会跑得快一些. 第一次测试最好运行得多一些, 毕竟 Clang/LLVM 是一个庞大的项目, 即使是很小的修改, 也有可能破坏掉一些名字看起来八竿子打不着的玩意. 

````bash
# 如果改的是 clang 相关的, 可以运行这个测试组
ninja -j16 check-clang
````

然后去摸摸鱼, 站一下, 看看远方, 等待测试出结果. 一般而言会失败个好几个测试, 这时候就可以对着输出结果去搜对应的测试文件来修了. (~~如果第一趟测试就都过了, 那您太强了. ~~)

举个例子, 这是我之前遇到过的一个错误输出:

````
Failed Tests (40):
  Clang :: ClangScanDeps/has_include_if_elif.cpp
  ... ... 省略类似输出 ... ...
  Clang :: ClangScanDeps/vfsoverlay.cpp
  Clang :: Index/annotate-macro-args.m
  Clang :: Index/c-index-getCursor-test.m
  Clang :: Index/get-cursor-macro-args.m
  Clang-Unit :: Tooling/./ToolingTests/CommentHandlerTest/BasicTest1
  Clang-Unit :: Tooling/./ToolingTests/CommentHandlerTest/BasicTest2
  ... ... 省略类似输出 ... ...
  Clang-Unit :: libclang/./libclangTests/LibclangReparseTest/PreprocessorSkippedRanges


Testing Time: 136.05s
  Skipped          :    35
  Unsupported      :  1681
  Passed           : 29796
  Expectedly Failed:    26
  Failed           :    40
````

首先, 测试有分回归测试和单元测试两种. 回归测试简单来讲就是 "过样例", 使用编译产物直接接受特定参数/输入, 对比输出是否合理. 单元测试简单来讲就是 "测函数", 直接编写代码检查特定函数的语义和返回值是否合理. 像上面这个例子, 像 `Clang :: ClangScanDeps/has_include_if_elif.cpp` 这种路径里面没有点的那就是回归测试, 对应的测试 (输入) 文件名就叫 `has_include_if_elif.cpp`. 那像 `Clang-Unit :: Tooling/./ToolingTests/CommentHandlerTest/BasicTest1` 这种名字里带点的就是单元测试, 紧跟着点后面的就是文件名, 然后再后面的就是测试模块. 比如说这个例子里, 放测试的文件名就叫 `ToolingTests` (一般是 `.cpp`), 然后直接在代码里搜索 `BasicTest1` 基本上就能搜到对应代码了.

### Clang 测试框架与环境

 > 
 > 本小节主要关注回归测试. 单元测试看起来跟写起来都比较显然, 就不细说了.

#### 一些 Perl 脚本的安装

我使用的发行版是 Fedora, 它在安装完 Perl 之后似乎并不能直接运行 clang 的全部测试, 会缺少一些特定的 Perl 库. 下面是我发现的缺失的 Perl 库:

````
````

#### llvm-lit 与 FileCheck

我们可以打开一个回归测试的文件, 比如说上面提到的 `has_include_if_elif.cpp` 来看看里面到底有什么:

````cpp
// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/has_include_if_elif2.cpp
// RUN: cp %s %t.dir/has_include_if_elif2_clangcl.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header2.h
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header3.h
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header4.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/has_include_if_elif.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -mode preprocess-dependency-directives | \
// RUN:   FileCheck %s
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -mode preprocess | \
// RUN:   FileCheck %s

#if __has_include("header.h")
#endif

#if 0
#elif __has_include("header2.h")
#endif

#define H3 __has_include("header3.h")
#if H3
#endif

#define H4 __has_include("header4.h")

#if 0
#elif H4
#endif

// CHECK: has_include_if_elif2.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h
// CHECK-NEXT: Inputs{{/|\\}}header2.h
// CHECK-NEXT: Inputs{{/|\\}}header3.h
// CHECK-NEXT: Inputs{{/|\\}}header4.h

// CHECK: has_include_if_elif2_clangcl.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h
// CHECK-NEXT: Inputs{{/|\\}}header2.h
// CHECK-NEXT: Inputs{{/|\\}}header3.h
// CHECK-NEXT: Inputs{{/|\\}}header4.h
````

首先, 从文件名就可以看出, 这是一个合法的 C++ 源代码文件. 文件头和尾写了很多看起来像是指导测试过程的命令. 实际上所谓运行一个回归测试, 就是在运行在这个测试文件里写着的各种命令, 然后按照这里写着的内容判断命令标准输出里的输出是否符合预期. 我们首先要认识两个基本工具:

* [`llvm-lit`](): 用于解析测试文件, 具体执行测试命令的工具
* [`FileCheck`](): 用于方便地对比两个输出的工具, 可以视作高级版 (支持诸如按行前缀匹配之类的) 的 `diff`

这两个工具都会从测试文件的注释里读取信息. 上面的例子中, 以 `RUN:` 开头的注释就是提供给 `llvm-lit` 的信息, 以 `CHECK` 开头的就是给 `FileCheck` 的信息. 具体的意思见下:

````cpp
//! 以这个前缀开头的注释就是我个人补充的

//! 下面是给 llvm-lit 的信息
//! 基本上就是带变量的多条 bash 命令的集合, llvm-lit 会读取命令, 替换以 % 开头的变量, 然后按顺序执行它们

//! 首先进行一些可能的清理. %t 代表一个可用的临时文件标识符, 一般一个文件只有一个, 所以要清理复用
// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir

//! %s 代表当前文件的文件名, %S 代表当前文件所在的文件夹. 可以看到为了运行测试, 
//! 它首先把测试文件和它所需要的, 放在代码里的其它文件先复制到了测试用临时文件夹里.
//! 这可能是为了避免测试在无意中覆写了给定的文件
// RUN: cp %s %t.dir/has_include_if_elif2.cpp
// RUN: cp %s %t.dir/has_include_if_elif2_clangcl.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header2.h
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header3.h
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header4.h

//! 在一些文件中, 我们很可能没办法写死一些路径. 比如说 compile_commands.json 里, 
//! 我们应该写下完整的编译命令, 而这要求我们写死一个绝对路径. 
//! 可我们在运行测试的时候是在一个临时文件夹里运行的, 没办法编写测试的时候就知道是哪个路径
//! 所以一般会有一个约定俗成的写法, 即在这种文件里用 DIR 代表运行测试时的实际目录, 
//! 然后在将其复制到测试目录之前, 先跑一遍 sed 把它替换掉, 得到真正可用的文件.
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/has_include_if_elif.json > %t.cdb

//! 接下来便是真正的测试了. 在测试时, llvm-lit 会默认优先把自己所在的目录当作 PATH 环境变量的第一个
//! 借此保证运行的是跟 llvm-lit 同目录下的编译产物, 而不是 (可能的) 系统里的那个正常的软件.
//! 当然有的时候你也可以看到有些测试文件使用 %clang 指代 clang 的, 这也是有的
//! FileCheck 从标准输入里接受测试命令的输出, 然后接受一个测试文件名当作参数, 
//! 从那个测试文件里读取 "判定为正确" 的规则. (也就是文件末尾的那些注释)
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -mode preprocess-dependency-directives | \
// RUN:   FileCheck %s
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -mode preprocess | \
// RUN:   FileCheck %s

//! 下面是我们想要测试的文件内容
//! 这个测试主要是测试 __has_include 的功能是否正常
#if __has_include("header.h")
#endif

#if 0
#elif __has_include("header2.h")
#endif

#define H3 __has_include("header3.h")
#if H3
#endif

#define H4 __has_include("header4.h")

#if 0
#elif H4
#endif

//! 下面是给 FileCheck 用的测试指令
//! CHECK 表示输出里是否存在后面跟着的内容
//! CHECK-NEXT 自然就是表示在检测到上一行的内容之后, 后面是否接着有这一行的内容
//! 可以看到我们在检测内容里可以表达一些常用的正则类似物, 
//! 比如这里就用了 {{/|\\}} 来让测试在 Linux 和 Windows 这两个采用两种路径分隔符的系统上都能正确识别内容
// CHECK: has_include_if_elif2.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h
// CHECK-NEXT: Inputs{{/|\\}}header2.h
// CHECK-NEXT: Inputs{{/|\\}}header3.h
// CHECK-NEXT: Inputs{{/|\\}}header4.h

// CHECK: has_include_if_elif2_clangcl.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h
// CHECK-NEXT: Inputs{{/|\\}}header2.h
// CHECK-NEXT: Inputs{{/|\\}}header3.h
// CHECK-NEXT: Inputs{{/|\\}}header4.h
````

要运行单个测试, 只需要:

````bash
# 在某个构建文件夹 (build) 里
bin/llvm-lit -vv ../clang/test/ClangScanDeps/has_include_if_elif.cpp         
````

开 `-vv` 可以让 `llvm-lit` 输出执行的具体命令和出错的具体位置. 它的输出我个人觉得有点难读, 建议自己细心好好体会一下 ;D.

如果你需要说以这个为输入, 调试 `clang-scan-dep` 的话, 那估计你是只能看懂这些命令在干什么并手动敲一下 `launch.json` 了. 我有计划做一个读取这种文件然后自动生成调试用 json 的小脚本, 不过目前仍在绝赞咕咕咕中...

当然, 上面提到的显然都只是 `llvm-lit` 和 `FileCheck` 的一小部分用法. 想要深入了解, 还得靠多看几个测试和看文档才行.

#### 工具专属的测试用 flag

Clang 有一些属于它自己的测试用 flag:

* `-cc1`
* `-fsyntax-only`
* `-verify`

#### 编写测试

### 发布前处理 1: Git message

终于, 测试全都通过了, 你也编写好了对应自己改动的测试, 是时候开始准备提交 patch 了! 不过, 在这之前, 你还需要为你的 patch 写一个详尽的说明...

 > 
 > 英文苦手 qwq, 不过建议与我同样对英文语法没啥自信的人可以先把自己写的东西去 [Grammarly](https://app.grammarly.com/) 过一遍, 至少做到没有正确性错误 (红色错误), 然后再口头读一遍是否通顺. 毕竟我觉得文法正确这一点就好像是码风优良一样的, 是很能体现出一个人对自己的 patch 上不上心的一件事.

LLVM 社区相对 Linux Kernel 来讲, 并没有非常地纠结 commit message 的格式. 一般而言, 做到下面的就可以了:

* **第一行必须是 `[子模块名] 简要概括` 的格式**
* **每一行不超过 72/80 个字符**, 最好是 72, 但是似乎 80 也是可以接受的.
* 描述一下你对代码作出修改的意图
* (可选) 描述一下你为什么要这么改
* 文末使用 "Link: xxx" 带上社区讨论的 review.llvm.org 或者说 GitHub issue 的连接

一般而言, 编出实际内容不成问题. 如果有疑问, 可以开 `git log` 翻一下历史. 不过这样做的话最好翻看个五十一百条左右的 commit 记录, 因为 LLVM 的 commit 数量很多, 一不小心可能手气差, 抽到的都是大佬交的 commit, 不一定就是适合新人的写法.

对于折行, 我使用的是 []() 这个 vscode 插件. 需要折的时候 Ctrl + P 跑一下就可以了. 如果懒得新开一个文件写 commit message 再复制粘贴到终端里, 可以 `export EDITOR="code --wait"`, 这样就可以直接在 VSCode 里写了.

### 发布前处理 2: Git 记录

一般来讲, 一个 patch 做个两三天的很正常. 所以当你终于完成了一个 patch, 打算发布时, main 分支可能已经跑出去好远了. 这时候就需要对 commit tree 做下面的事情:

下面是对应的命令:

````bash
patch> git log # remeber your commit hash
patch> git checkout main
main> git pull
main> git checkout tmp
tmp> git cherrypick ${YOUR_COMMIT_HASH}
tmp> git branch -M patch
patch> # finish !
````

另外, 最好在提交之前再检查一遍 clang-format:

 > 
 > Fedora 上需要额外装: `sudo dnf install git-clang-format`

````bash
patch> git clang-format
# 如果有修改的提示的话
patch> git commit --amend 
````

### 发布前处理 3: 论坛

最后, 就是要将你的 patch 发布到 Phabricator 了. 首先, 你需要安装 [Arcanist](https://secure.phabricator.com/book/phabricator/article/arcanist/), 在 Phabricator 上发帖用的命令行工具. 随后, 你就可以使用下面的命令将你目前的修改打包并发成一个串:

````bash
arc diff --draft
````

切记要加 `--draft` 参数, 让它发布一个草稿而不是直接就发出去. 如果它叫你填什么 Reviewer, Subscriber, **不要在这里填, 直接留空**. 到最后它会给你一个类似 `https://reviews.llvm.org/D137343` 这样的链接, 这就是你的 patch (现在叫 Differential 了) 的串的链接了.

这个工具有点年久失修了, 经过它的手, 你的 commit message 可能在 Phabricator 上看起来面目全非, 并且在这里填 Reviewer 和 Subscriber 也不方便, 没有自动补全. 所以我们需要一个草稿, 然后上 web 网页端再润色一下格式并挑选合适的 Reviewer 和 Subscriber.

````
运行 clang-format
    或者开启 format at save
装 arc
arc diff --draft
加 reviewer
    如何加
    加谁
````
