
 > 
 > The QFormLayout class manages forms of input widgets and their associated labels.

简单来说, 就是两列多行的, label + widget 的组合.

例子:

````cpp
QFormLayout *formLayout = new QFormLayout;
// & 表示对齐的位置
formLayout->addRow(tr("&Name:"), nameLineEdit);
formLayout->addRow(tr("&Email:"), emailLineEdit);
formLayout->addRow(tr("&Age:"), ageSpinBox);
setLayout(formLayout);
````
