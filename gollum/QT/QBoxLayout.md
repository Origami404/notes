
 > 
 > The QBoxLayout class lines up child widgets horizontally or vertically.

它将组件的区域划分为一个个盒子 (一般大小相同), 然后把子组件放进盒子里. 盒子可以水平也可以竖直排列 (通过 `orientation` 更改), 盒子的排列顺序也可以设置 (`direction`, 比如可以逆序右到左这样子).

一般我们会直接使用 QHBoxLayout 或 QVBoxLayout.

如果一个 QBoxLayout 不是一个顶层 layout, **必须在对它做任何操作之前将其加入到父 layout 中**.

常用方法:

* `addWidget`: 加入子组件
* `addSpacing`: 加入一个空白盒子
* `addStretch`: 加入一个可伸缩的空白盒子
* `addLayout`: 加入一个子 layout

上面的 add 系方法都有对应的 insert 系方法, 后者可以指定插入的位置.

若要调整间距, 可以使用下面的方法:

* `setContensMargins`: 设置自己这个 layout 对四边留下的空白大小
* `setSpacing`: 设置每个子盒子对邻居的空隙大小

子组件可以被移除 (`remove`) 或隐藏 (`hide`), 后者可以快速地被重新显示 (`show`).
