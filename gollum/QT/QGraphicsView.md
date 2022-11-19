提供视图, 负责显示 [QGraphicsScene](QGraphicsScene.md). 一个 Scene 可以链接到多个视图.

````cpp
QGraphicsScene scene;
myPopulateScene(&scene);
QGraphicsView view(&scene);
view.show();
````

缩放与旋转:

````cpp
class View : public QGraphicsView
{
Q_OBJECT
    ...
public slots:
    void zoomIn() { scale(1.2, 1.2); }
    void zoomOut() { scale(1 / 1.2, 1 / 1.2); }
    void rotateLeft() { rotate(-10); }
    void rotateRight() { rotate(10); }
    ...
};
````
