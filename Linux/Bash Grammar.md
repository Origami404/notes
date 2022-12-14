## 简单指令

 > 
 > A simple command is a sequence of optional variable assignments followed by blank-separated words and redirections, and terminated by a control operator. 

````
<simple-command> -> <variable-assignment>p m s j* <blank-sparated words> <redirection>* <control-operator>
````

 > 
 > The first word specifies the command to be executed, and is passed as argument zero. The remaining words are passed as arguments to the invoked command.

## 语法 Overview

使用 `A::B` 指示 "用 B 分隔的 A 列表"

````antlr
nls: newlines, newline list, 指一个或多个 newline
sop: separator_op, 分割不同 bash 指令用的操作符, 可以是 '&' 或 ';'

complete_cmd  :  list sop?
list          :  and_or::sop
pipeline      :  "!"? cmd::"|"

cmd: simple_cmd | compound_cmd redirect* | func_def

compound_cmd
    : '{' list '}'            # brace group
    | '(' list ')'            # subshell
    | for | case | if | while | until

redirect: 详见下面的章节

func_def: name '(' ')' compound_cmd redirect?
````

## 重定向相关 (Redirection)

参考: <https://www.gnu.org/software/bash/manual/html_node/Redirections.html>

**重定向带顺序的!!! 由后往前执行!!!!** `2>&1 >f` 会先 `>f`, 然后才 `2>&1`, 四舍五入没用. 得写 `>f 2>&1`

下面用 `n`, `nn` 代表数字 (文件描述符), `m` 代表数字或 `-`, `f` 代表文件名, `T` 代表任意文本, `[]` 中代表可选

* `[n]<f`: 重定向输入 `n(=0)` 为文件 `f`
* `[n]>[|]f`: 重定向输出 `n(=1)` 为文件 `f`, 有 `|` 时不覆盖文件
* `[n]>>f`: 重定向输出, 但是附加模式
* `&>f`, `>&f`: 重定向 `1` 和 `2` 到文件 `f`, 优先选择第一种, 等价于 `>f 2>&1` (注意顺序)
* `&>>f`: 重定向 `1`, `2`, 但是附加模式, `>>f 2>&1`
* `[n]<<[-]d ...TEXT d`: here-document, 向 `n(=0)` 里送 `...TEXT`, 带 `-` 则忽略 `...TEXT` 中的行首 Tab
* `[n]<<< ...TEXT`: here-string
* `[n]<&m`: 将 `n(=0)` 设置为 `m` 的复制; 若 `m=-`, 则关闭 `n`
* `[n]>&m`: 将 `m` 设置为 `n(=1)` 的复制; 若 `m=-`, 则关闭 `n`
* `[n]<&nn-`, 将 `nn` "移动" 到 `n(=0)` (也就是复制到, 再关闭)
* `[n]>&nn-`, 将 `n(=1)` "移动" 到 `nn`

一些特殊的文件:

* `/dev/fd/n`: 文件描述符 `n`
* `/dev/stdin`, `/dev/stdout`, `/dev/stderr`
* `/dev/tcp/host/port`: 建立 TCP 链接
* `/dev/udp/host/port`: 建立 UDP 链接

## Bash 表达式展开过程

参考: <https://www.gnu.org/software/bash/manual/html_node/Shell-Expansions.html>

* 大括号展开 (brace expansion)
* 波浪号展开 (tilde expansion)
* 参数展开 (parameter expansion)
* 命令替换 (command substitution)
* 算术展开 (arithmetic expansion)
* 单词切分 (word splitting)
* 文件名展开 (filename expansion)
* (可选) 进程替换 (process substitution)
* 引号删除 (quote removal)

### 大括号展开

 > 
 > 多选

类似文件名展开, 但是文件名展开要求文件存在, 而大括号展开不需要.

````bash
$ echo a{d, c, b}e
ade ace abe
````

大括号内还可以写数字/字符+增长, 比如 `{x..y[..incr]}`

大括号展开是严格 "词法" 的, 它不会对任何特殊字符作出 "反应", 包括大括号自己 (

````bash
chown root /usr/{ucb/{ex,edit},lib/{ex?.?*,how_ex}}
````

### 波浪号展开

 > 
 > 快速访问[目录栈](https://www.gnu.org/software/bash/manual/html_node/The-Directory-Stack.html)与 HOME

* `~`: `$HOME`
* `~/foo`: `$HOME/foo`
* `~fred/foo`: fred 的 HOME 下的 foo 文件夹
* `~+/foo`: `$PWD/foo`
* `~-/foo`: `${OLDPWD-'~-'}/foo`, 但是看起来好像跟 `$OLDPWD/foo` 没什么不同, 可能那个大括号是删掉斜杠的
* `~N`, `~+N`: `dirs +N`, 其实 `~+` 也可以看作 `N=0` 的情况
* `~-N`: `dirs -N`, 其实 `~-` 也可以看作 `N=1` 的情况

### 参数展开

* `${!p*}`, `${!p@}`: 间接展开, 先展开名字前缀为 `p` 的变量, 把展开后的东西当变量名再展开一次
* `${p:-default}`: 默认值
* `${p:=default}`: 空的时候不但用默认值代替, 还会赋值
* `${p:?prompt}`: 空的时候往 stderr 打 prompt
* `${p:+prompt}`: 不空的时候往 stderr 打 prompt
* `${p:offset}`, `${p:offset:length}`: 切片, 从 `[offset, offset + lenght)`, 从 0 开始, 负数反之. 负数 offset 负号前要打空格, 否则会跟 `:-` 冲突. 如果 p 是 `@` 或是 `*`, 那么这个切片对位置参数数组做切片
* `${#p}`: 获得 p 展开后的字符长度
* `${p@op}`: 对 p 展开后做一些变换:
  * `U`: 转全大写
  * `u`: 转首字母大写
  * `L`: 转全小写
  * `Q`: 转成 "quoted" 的形式, 即还可以被 input 读回去的形式
  * `E`: 展开后带一些反斜杠转义, 使结果成为有效的变量名, 可以丢进 `$'...'`
  * `P`: 展开后, 使结果成为有效的 Prompt
  * `A`: 展开成 `p=$p` 的格式
  * `K`: 跟 `A` 差不多, 但是对数组会展开成键值对赋值的形式
  * `a`: 展开成 p 的属性
  * `k`: 跟 `K` 差不多, 但是展开后的东西会过一边单词切分

还有一堆跟[模式匹配](https://www.gnu.org/software/bash/manual/html_node/Pattern-Matching.html)有关的变量展开, 标志是大括号里有 `#%/^,` 的

### 命令替换

`$(command)` 和 ``command``. 特别地, `$(cat file)` 可以被替换为更快的 `$(< file)`. 命令替换可以叠加, 如果使用两个反引号引住命令替换, 替换后的内容将不会经过内层的单词切分和文件名展开.

### 算术展开

`$(( expr ))`. 里面的单词会经历参数展开, 变量替换和引号消除, 可以嵌套. 合法的算式可以看这里 [Shell Arithmetic](https://www.gnu.org/software/bash/manual/html_node/Shell-Arithmetic.html)

### 进程替换

* `<(cmd-list)`: 执行 `cmd-list`, 其输出会被写入到某个文件里, 然后这个文件的路径被当成展开结果
* `>(cmd-list)`: 会被展开成一个文件路径, 然后往该路径里的任何写入会被当成 cmd-list 的标准输入

参考: [SO](https://unix.stackexchange.com/a/27346/546053)

常见使用:

````bash
# 这样不行, 因为 bash 的 stdin 是读给脚本当 stdin 的, 不是用来读脚本的
curl -o - http://example.com/script.sh | bash

# 这样可以, 下载下来的脚本被存到某个文件里, 然后 bash 可以接受一个文件名作为参数执行那个文件
bash <(curl -o - http://example.com/script.sh)
````

````bash
# 查找所有我不够权限看的文件
# 这里把标准错误写到了一个文件里, 这个文件里的内容会被当成 sed 的标准输入
(ls /proc/*/exe >/dev/null) 2> >(sed -n \
  '/Permission denied/ s/.*\(\/proc.*\):.*/\1/p' > denied.txt )
````

### 单词切分

将 `$IFS` 当作单词切分的分隔符, 然后切之. 需要注意的是, 如果没有进行过扩展, 那么单词切分也不会被进行.

 > 
 > When a quoted null argument appears as part of a word whose expansion is non-null, the null argument is removed.

这意味着 `-d''` 并不会是 `-d ''`, 而只是 `-d`. 

### 文件名扩展

参见 [glob](https://en.wikipedia.org/wiki/Glob_(programming)).

### 引号移除

移除所有引号并进行转义.

## 完整 Posix Shell 语法

From: <https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_10>

````
/* -------------------------------------------------------
   The grammar symbols
   ------------------------------------------------------- */
%token  WORD
%token  ASSIGNMENT_WORD
%token  NAME
%token  NEWLINE
%token  IO_NUMBER
  
/* The following are the operators (see XBD [_Operator_](https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap03.html#tag_03_260))
   containing more than one character. */

%token  AND_IF    OR_IF    DSEMI
/*      '&&'      '||'     ';;'    */
  
%token  DLESS  DGREAT  LESSAND  GREATAND  LESSGREAT  DLESSDASH
/*      '<<'   '>>'    '<&'     '>&'      '<>'       '<<-'   */
  
%token  CLOBBER
/*      '>|'   */
  
/* The following are the reserved words. */
  
%token  If    Then    Else    Elif    Fi    Do    Done
/*      'if'  'then'  'else'  'elif'  'fi'  'do'  'done'   */
  
%token  Case    Esac    While    Until    For
/*      'case'  'esac'  'while'  'until'  'for'   */
  
/* These are reserved words, not operator tokens, and are
   recognized when reserved words are recognized. */
  
%token  Lbrace    Rbrace    Bang
/*      '{'       '}'       '!'   */
  
%token  In
/*      'in'   */
  
/* -------------------------------------------------------
   The Grammar
   ------------------------------------------------------- */
%start program
%%
program          : linebreak complete_commands linebreak
                 | linebreak
                 ;
complete_commands: complete_commands newline_list complete_command
                 |                                complete_command
                 ;
complete_command : list separator_op
                 | list
                 ;
list             : list separator_op and_or
                 |                   and_or
                 ;
and_or           :                         pipeline
                 | and_or AND_IF linebreak pipeline
                 | and_or OR_IF  linebreak pipeline
                 ;
pipeline         :      pipe_sequence
                 | Bang pipe_sequence
                 ;
pipe_sequence    :                             command
                 | pipe_sequence '|' linebreak command
                 ;
command          : simple_command
                 | compound_command
                 | compound_command redirect_list
                 | function_definition
                 ;
compound_command : brace_group
                 | subshell
                 | for_clause
                 | case_clause
                 | if_clause
                 | while_clause
                 | until_clause
                 ;
subshell         : '(' compound_list ')'
                 ;
compound_list    : linebreak term
                 | linebreak term separator
                 ;
term             : term separator and_or
                 |                and_or
                 ;
for_clause       : For name                                      do_group
                 | For name                       sequential_sep do_group
                 | For name linebreak in          sequential_sep do_group
                 | For name linebreak in wordlist sequential_sep do_group
                 ;
name             : NAME                     /* Apply rule 5 */
                 ;
in               : In                       /* Apply rule 6 */
                 ;
wordlist         : wordlist WORD
                 |          WORD
                 ;
case_clause      : Case WORD linebreak in linebreak case_list    Esac
                 | Case WORD linebreak in linebreak case_list_ns Esac
                 | Case WORD linebreak in linebreak              Esac
                 ;
case_list_ns     : case_list case_item_ns
                 |           case_item_ns
                 ;
case_list        : case_list case_item
                 |           case_item
                 ;
case_item_ns     :     pattern ')' linebreak
                 |     pattern ')' compound_list
                 | '(' pattern ')' linebreak
                 | '(' pattern ')' compound_list
                 ;
case_item        :     pattern ')' linebreak     DSEMI linebreak
                 |     pattern ')' compound_list DSEMI linebreak
                 | '(' pattern ')' linebreak     DSEMI linebreak
                 | '(' pattern ')' compound_list DSEMI linebreak
                 ;
pattern          :             WORD         /* Apply rule 4 */
                 | pattern '|' WORD         /* Do not apply rule 4 */
                 ;
if_clause        : If compound_list Then compound_list else_part Fi
                 | If compound_list Then compound_list           Fi
                 ;
else_part        : Elif compound_list Then compound_list
                 | Elif compound_list Then compound_list else_part
                 | Else compound_list
                 ;
while_clause     : While compound_list do_group
                 ;
until_clause     : Until compound_list do_group
                 ;
function_definition : fname '(' ')' linebreak function_body
                 ;
function_body    : compound_command                /* Apply rule 9 */
                 | compound_command redirect_list  /* Apply rule 9 */
                 ;
fname            : NAME                            /* Apply rule 8 */
                 ;
brace_group      : Lbrace compound_list Rbrace
                 ;
do_group         : Do compound_list Done           /* Apply rule 6 */
                 ;
simple_command   : cmd_prefix cmd_word cmd_suffix
                 | cmd_prefix cmd_word
                 | cmd_prefix
                 | cmd_name cmd_suffix
                 | cmd_name
                 ;
cmd_name         : WORD                   /* Apply rule 7a */
                 ;
cmd_word         : WORD                   /* Apply rule 7b */
                 ;
cmd_prefix       :            io_redirect
                 | cmd_prefix io_redirect
                 |            ASSIGNMENT_WORD
                 | cmd_prefix ASSIGNMENT_WORD
                 ;
cmd_suffix       :            io_redirect
                 | cmd_suffix io_redirect
                 |            WORD
                 | cmd_suffix WORD
                 ;
redirect_list    :               io_redirect
                 | redirect_list io_redirect
                 ;
io_redirect      :           io_file
                 | IO_NUMBER io_file
                 |           io_here
                 | IO_NUMBER io_here
                 ;
io_file          : '<'       filename
                 | LESSAND   filename
                 | '>'       filename
                 | GREATAND  filename
                 | DGREAT    filename
                 | LESSGREAT filename
                 | CLOBBER   filename
                 ;
filename         : WORD                      /* Apply rule 2 */
                 ;
io_here          : DLESS     here_end
                 | DLESSDASH here_end
                 ;
here_end         : WORD                      /* Apply rule 3 */
                 ;
newline_list     :              NEWLINE
                 | newline_list NEWLINE
                 ;
linebreak        : newline_list
                 | /* empty */
                 ;
separator_op     : '&'
                 | ';'
                 ;
separator        : separator_op linebreak
                 | newline_list
                 ;
sequential_sep   : ';' linebreak
                 | newline_list
                 ;
````
