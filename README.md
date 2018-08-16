## Focal loss
- focal loss with multi-label implemented in keras.
- reference to paper : [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)

## Usage
- use like : 
> firstly, you should get a list which contains each class number, like classes_nu=[1,2,3] means index_0 class have 1 pic, index_1 class have 1 pics.
> then, use the focal loss function like below:
> model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss=[focal_loss(classes_num)], metrics=['accuracy'])

## blog of focal loss
- [focal loss论文笔记(附基于keras的多类别focal loss代码)](https://blog.csdn.net/qq_42277222/article/details/81711289)
