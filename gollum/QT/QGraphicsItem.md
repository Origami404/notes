有一些现成的子类: rectangles ([QGraphicsRectItem](https://doc.qt.io/qt-5/qgraphicsrectitem.html)), ellipses ([QGraphicsEllipseItem](https://doc.qt.io/qt-5/qgraphicsellipseitem.html)) and text items ([QGraphicsTextItem](https://doc.qt.io/qt-5/qgraphicstextitem.html)).

一个 item 可以处理:

* 鼠标点击, 移动, 释放, 双击, 悬浮, 滚轮, 右键
* 键盘输入焦点与击键
* 拖动
* 分组(?) (Grouping, both through parent-child relationships, and with [QGraphicsItemGroup](https://doc.qt.io/qt-5/qgraphicsitemgroup.html))
* 碰撞检测
