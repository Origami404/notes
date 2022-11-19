## 参考

* [Getting Started with the LLVM System](https://llvm.org/docs/GettingStarted.html)
* [Getting Started: Building and Running Clang](https://clang.llvm.org/get_started.html)
* 来自 [inclyc](https://github.com/inclyc) 的指点

## 源码下载

源码仓库目前 (2022/10/25) 大约有 3.3G 大, 从 GitHub 下可能有点慢. 清华提供了 [镜像仓库](https://mirrors.tuna.tsinghua.edu.cn/help/llvm-project.git/):

````bash
# 普通解析
git clone https://mirrors.tuna.tsinghua.edu.cn/git/llvm-project.git

# IPv6
git clone https://mirrors6.tuna.tsinghua.edu.cn/git/llvm-project.git
````

推荐使用 IPv6 下载, 因为据清华那边的小道消息, IPv6 不卡 IO. 实测能下得非常快 (教育网 IPv6 下大概 20-40 MB/s). 不过下下来之后可能要替换一下 `origin`, 并再跟官方仓库同步一下:

````bash
git remote remove origin
git remote add origin https://github.com/llvm/llvm-project.git
git branch --set-upstream-to=origin/main main
git pull
````

## 编译命令

inclyc 分享的 CMake 参数, 可加速编译流程, 极大减少内存占用, 极大减少编译产物大小, 加快 Debug 的启动. 我稍微包装了一下做成了脚本放在了 `local/bin` 里.

````bash
#!/usr/bin/bash

build_mode=${1:-'release'}
llvm_enable_projects=${2:-'clang'}

cmake -DBUILD_SHARED_LIBS=On \
    -DCMAKE_BUILD_TYPE=${build_mode} \
    -DLLVM_APPEND_VC_REV=Off \
    -DLLVM_ENABLE_LLD=On \
    -DLLVM_ENABLE_PROJECTS="${llvm_enable_projects}" \
    -DLLVM_LINK_LLVM_DYLIB=Off \
    -DLLVM_OPTIMIZED_TABLEGEN=On \
    -DLLVM_CCACHE_BUILD=TRUE \
    -DLLVM_USE_SPLIT_DWARF=On \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
    -DCMAKE_C_COMPILER=${CC:-"clang"} \
    -DCMAKE_CXX_COMPILER=${CXX:-"clang++"} \
    -G Ninja \
    -DLLVM_TARGETS_TO_BUILD="host;" \
    ${@:3}
````

主要参数解释:

* `BUILD_SHARED_LIBS=On`: 编译动态库而不是静态链接, 可以缩短链接时间 + 减少编译产物大小
* `LLVM_ENABLE_LLD=On`: 使用 lld 而不是 ld 来链接. lld 是 LLVM 里用来原地替换 ld 的链接器, 据说链接时能省不少内存.
  * From inclyc: "最好不要用 mold, 它容易 fork 出过多进程然后卡死调度"
* `LLVM_OPTIMIZED_TABLEGEN=On`: 在 Debug 模式下使用优化的 [TableGen](https://llvm.org/docs/TableGen/). TableGen 是 LLVM 后端用于描述平台信息相关的 DSL 和小工具. 一般它不会出 bug, 所以即使在 Debug 模式下也可以使用对它的优化.
* `LLVM_CCAHCHE_BUILD=TRUE`: 使用 [Ccach](https://ccache.dev/). 一般我们用 Makefile 之类的构建工具的话, 它会自动帮我们检测源文件时间和编译产物时间来决定要不要重新编译. 但是这种选择经常跟项目里各种各样的配置混杂在一起, 导致我们经常需要 `make clean && make` 重新来一趟干净的编译. Ccache 将 "编译缓存" 这件事单独拉出来, 在确保安全的情况下, 即使执行了 `make clean` 再 `make`, 也可以复用之前的编译结果, 加快编译进度. 
* `LLVM_USE_SPLIT_DWARF=On`: `DWARF` 是一种调试信息格式, 这个配置可以分离可执行文件与调试信息, 在具体调试时再按需加载符号信息, 能提高调试启动速度.
* `CMAKE_EXPORT_COMPILE_COMMANDS=TRUE`: 生成 `compile_commands.json`. (不会有人不用 [Clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) 来在 VSCode 上看 LLVM 源码吧)
* `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`: 这两个必须指定, 似乎 CMake 并不会直接从 `CC`/`CXX` 环境变量中读取编译器配置. 使用 clang(++) 编译的话比 gcc(++) 要快一些, 还省点内存.
* `-G Ninja`: 使用 Ninja 作为 Makefile 的替代. 一般而言 Ninja 会比 Makefile 快且轻量, 而且输出比较友好.
* `LLVM_APPEND_VC_REV=Off`: 不要在 `LLVM_VERSION_STRING` 里附加小修订版本的版本字符串. 因为大部分小修订版本并不会影响大部分头文件, 如果每次小修订都修改这个字符串的话, 很多头文件就都要在每次 commit 之后重新编译, 浪费编译时间. 详见 [D37272](https://reviews.llvm.org/D37272)
* `LLVM_LINK_LLVM_DYLIB=Off`: 详见 [LLVM CMake 文档](https://llvm.org/docs/CMake.html), 虽然这个 flag 在当前版本 (`6951cec`) 默认就是关闭的, 并且不能跟 `BUILD_SHARED_LIBS` 一起 On

另外根据 inclyc 的推荐, 可以同时编译 Release 和 Debug 两个版本, Debug 用于 Debug, Release 用于跑单元测试 (Release 跑得快). 只要开两个文件夹 (`build-debug` 和 `build-release`) 就可以了. 另外要编译 Clang 的话记得将 clang 加入到 `LLVM_ENABLE_PROJECTS` 里, 然后 `ninja` 构建目标要选 clang.

````bash
mkdir build && cd build
cmake ${flags} -DLLVM_ENABLE_PROJECTS="clang" ../llvm
ninja -j16 clang
````

根据我的测试, 各种资源占用大概如下:

* 不使用任何 flag, 使用 clang/clang++/ld 编译:
  * 配置: 台式 i7-10700 + 16G 内存 + SSD
  * 编译时 `-j16` 最高内存占用大概 6G
  * 链接时 `-j4` 最高内存占用大概 10G
  * Debug 构建产物 24G
  * 编译时间大约 20-30 min
* 使用上述 flag:
  * 配置: 笔记本 i7-11700H + 32G 内存 + SSD
  * 编译与链接 `-j16` 最高内存占用大概 6G
  * Debug 构建产物 7.3G
  * Release 构建产物 500M
  * Debug 编译时间约 15-20 min
  * Release 编译时间约 10-15min

## 调试命令

安装 [CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb), 然后直接点调试, 填一下 `.vscode/launch.json` 就可以了:

````json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug",
            // 写编译出的 clang 可执行文件路径
            "program": "${workspaceFolder}/build-debug/bin/clang",
            "args": [
                // 实测必须使用 -c 参数, 否则好像调试进不去 clang 内部的函数
                "-c",
                "${workspaceFolder}/local/test/fpow.c",
            ],
            "cwd": "${workspaceFolder}"
        }
    ]
}
````

根据 inclyc 的指点, 初入 Clang 的话, 应该在 `clang/lib/Parse/Parser.cpp:611` 的函数 `Parser::ParseTopLevelDecl` 开头打断点进行单步调试, 便能摸清 Clang 的大体流程.

## 小技巧

### 如何在项目文件夹里放置私人文件?

大项目一般都会把 `.gitignore` 放进仓库里面, 我们想要存放一些我们自己用的脚本的时候就不可能说自己建一个文件夹然后直接给写进 `.gitignore` 里面, 这样会污染 `.gitignore`. 正确的做法是编辑 `.git/info/exclude`, 将它当作 "私人 `.gitignore`" 使用. 

比如说要把个人文件全都放在项目下的 `local` 里话那就:

````bash
echo 'local' >> .git/info/exclude

# 一般用 VSCode 的话这个也要丢进去的
echo '.vscode' >> .git/info/exclude
````
