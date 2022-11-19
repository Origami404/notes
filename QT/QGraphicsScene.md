一个场景可以提供:

* 是 [QGraphicsItem](QGraphicsItem.md) 的容器
* 内部对象维护接口
* 在元素之间传递事件
* 管理对象状态: 如选择与焦点触发
* 提供 (无变换的) 渲染 (通过各个对象的 `print` 方法)

````cpp
QGraphicsScene scene;

# addRect == 构造 Rect + addItem
QGraphicsRectItem *rect = scene.addRect(QRectF(0, 0, 100, 100));

# 访问对应位置的 item
QGraphicsItem *item = scene.itemAt(50, 50, QTransform());

# 当 GraphicsScene 接受到鼠标点击, 会自动把鼠标点击事件送给对应位置的元素
# 可以在代码里手动对元素发送特定事件, 如 selectedItems, setFocus 等

# 使用 render() 方法可以渲染特定区域
````
