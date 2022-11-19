QT 是非常常用的图像框架, 虽然我并不喜欢写 GUI, 但是为了生活...

Index: <https://doc.qt.io/qt-5/qtwidgets-module.html>

QT 核心概念:

* [QWidget](QWidget.md): 组件, 界面的单元
* [QLayout](QLayout.md): 布局, 确定组件位置
* [QPainter](QPainter.md): 绘制的基础
* 

我用过的其他部分:

* [QT Graphics View Framework](QT%20Graphics%20View%20Framework.md): 任意 2D 对象自由分布

一般而言, 使用 QT 写的程序都是遵循 "M-VC" 模式的 -- 对一个事物有两个类来模拟, 分别是代表那个事物本身的类 (Model), 代表那个事物在屏幕上显示的内容的类 (View), 同时也用来接受 View 上的事件并将其反映到 Model 中. 

### 常用的末端组件

* [QMainWindow](QMainWindow.md) 和其配套措施. 基本上所有程序都要有一个主窗口, 而这个组件便是最方便的绘制主窗口的方式.
