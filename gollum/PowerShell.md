主要记录学习该教程时的笔记: <https://devblogs.microsoft.com/scripting/powershell-for-programmers-a-quick-start-guide/>

## CMDLET

`cmdlet` (`command-let`), 基本上就是函数. 需要执行的命令一般都是 cmdlet.

* PowerShell **大小写不敏感**
* Cmdlet 总是 `<动词>-<名词>` 的形式
* Cmdlet 的参数叫 “Parameters”
* 要给 cmdlet 传递参数, 默认行为应该是使用具名参数, 而不是像正常的语言一样默认用顺序; 但是对一些特殊的命令也可以用顺序
  * `Get-Service("alg")` (×)
  * `Get-Service -Name "alg"` (√)
* 有的参数没有值, 称为 `Switch Parameter`
* `man`: `get-command -name "get-service" -Syntax`
* 图形化版详细文档: `Get-help "Get-Service" -showWindow`
* 顺路附带的 IDE: `powershell_ise.exe`

## 变量

````powershell
#easy and best practice way to create most variables
$foo = 5
$bar = get-service alg

#Powershell does not care what you put in your variables
$foo = 5
$foo

$foo = get-service alg
$foo

$foo = Get-Process
$foo
````

变量也可以是强类型的:

````powershell
[int]$foo = 5
$foo

#notice PowerShell will add an auto-convertion step before erroring
$foo = 5.5 #turning a double into an int is easy
$foo
$foo = "hello world" #take a look at the error

# 类型转换 (Cast)
#its a string
$foo = "5"
$foo.GetType().Name

#its an int
$foo = [int]"5"
$foo.GetType().Name

#user input explicit convertion without strong typing
$input = [int](read-host "give me a number:")
````

## 对象

````powershell
#Accessing the properties and methods
$service = get-service alg
$service.DisplayName
$service.GetType()

#you don't need them in variables for quick one-offs
(get-service Winmgmt).DependentServices

# 访问静态成员要用 `::`
[math]::Round(57575 / 1kb)
[math]::Max(5,2)

# 构造对象
$stack = new-object -TypeName System.Collections.Stack
$stack.push(1)
$stack.push(2)
$stack.push(3)
$stack.pop()

# isinstance
$service = get-service alg
$service -is [system.serviceprocess.servicecontroller]
````

Python 的 `__dir__`:

````powershell
#for whatever reason they didn't make -inputobject a positional parameter
$service = get-service alg
Get-Member -InputObject $service

#We can utilize the pipe though, and get member has a GM alias.
$service | Get-Member
get-process powershell_ise | GM

#This will force all the properties to come out in a list with their names and values
$service | Format-List -Property *
get-process powershell_ise | FL * #Nice shortcuts
````

## 条件

````powershell
# 不能使用运算符 == != 之类的
# 必须使用 -eq -ne -gt -ge -lt -le -not -or -and -xor
If (1 -eq 1) {
    write-host "Hello World" -ForegroundColor Cyan
}

# 字符串特有的运算符 -like -notlike, 作用于通配符上
# wild cards enabled
If ("hello" -like "h*") {
    write-host "equal!" -ForegroundColor green
} else {
    write-host "not equal!" -ForegroundColor red
}

# 正则表达式特有的运算符 -match -notmatch
#Check pattern
If ("I've got 1 digit inside!" -match "\d") {
    write-host "match!" -ForegroundColor green
    #cool way to see *what* matched (will only pull the first match)
    $matches[0]
}

# 判断类型使用 -is -isnot
# 判断值是否在 List 内用 -in -contains -notin -notcontains
# 如果需要大小写敏感的判断, 可以使用 -ceq -cne -clike -cnotlike 等带 c 前缀的运算符
````

## 引号

* 单引号是不展开变量的, 双引号是展开的
* 一般来讲直接展开变量只会得到它的类型, 想要得到其值必须得用 `$(表达式)` 包起来

````powershell
$service = get-service alg

"Service: $service"         # Service: System.ServiceProcess.ServiceController
"Service: $service.Name"    # Service: System.ServiceProcess.ServiceController.Name
"Service: $($Service.Name)" # Service: alg
````
