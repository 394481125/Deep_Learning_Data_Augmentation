# Deep_Learning_Data_Augmentation
 深度学习数据增广方法

## 方案一：使用工具imgaug增强数据

### 1、imgaug的介绍

​		imgaug是用于机器学习实验中图像增强的库。它支持广泛的扩充技术，可以轻松地组合它们并以随机顺序或在多个CPU内核上执行它们，具有简单而强大的随机界面，不仅可以扩充图像，还可以扩充关键点/地标，边界框，热图和分段图。

![大量扩充](https://imgaug.readthedocs.io/en/latest/_images/heavy.jpg)

### 2、imgaug的安装与卸载

​		该库使用必须安装的python。支持Python 2.7、3.4、3.5、3.6、3.7和3.8。

#### 2.1、在Anaconda中安装

```python
conda config --add channels conda-forge
conda install imgaug
```

使用扩充器`imgaug.augmenters.imgcorruptlike`需要安装`imagecorruptions`

```python
pip install imagecorruptions
```

#### 2.2、在`pip`中安装

```python
pip install imgaug
```

遇到问题需要安装`Shapely`

```python
pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio
pip install --no-dependencies imgaug
```

最新版本安装：

```python
pip install git+https://github.com/aleju/imgaug.git
```

使用扩充器`imgaug.augmenters.imgcorruptlike`需要安装`imagecorruptions`

```python
pip install imagecorruptions
```

#### 2.3、卸载

在Anaconda中卸载

```
conda remove imgaug
```

在pip中卸载

```
pip uninstall imgaug
```

#### 2.4、使用方法

- [API介绍及参考](https://blog.csdn.net/zong596568821xp/article/details/83105700)
- [API使用及参考](https://blog.csdn.net/u012897374/article/details/80142744)
- [官方文档](https://imgaug.readthedocs.io/en/latest/index.html)
- [我的colab实现地址](https://colab.research.google.com/drive/1rkrfWJuWIkWqCaCGWp43N2DTbt3nZjTe?usp=sharing)

#### 2.5、官方ipynb使用方法介绍

- [A01 - Load and Augment an Image](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [A03 - Multicore Augmentation](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [B01 - Augment Keypoints (aka Landmarks)](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [B02 - Augment Bounding Boxes](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [B03 - Augment Polygons](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [B06 - Augment Line Strings](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [B04 - Augment Heatmaps](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [B05 - Augment Segmentation Maps](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [C01 - Using Probability Distributions as Parameters](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [C02 - Using imgaug with more Control Flow](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [C03 - Stochastic and Deterministic Augmentation](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
- [C04 - Copying Random States and Using Multiple Augmentation Sequences](https://github.com/aleju/imgaug-doc/tree/master/notebooks)

## 方案二：使用工具opencv增强数据





## 方案三：tensorflow库的ImageDataGenerator增强数据





## 方案四：torchvision库的transforms增强数据







## 参考资料：

[imgaug](https://github.com/aleju/imgaug)

[imgaug-doc](https://github.com/aleju/imgaug-doc)

[imgaug官方使用手册](https://imgaug.readthedocs.io/en/latest/index.html)

[图片数据不够快来试试使用imgaug增强数据](https://xiulian.blog.csdn.net/article/details/105547204)

[imgaug数据增强神器：增强器一览](https://blog.csdn.net/lly1122334/article/details/88944589?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase)

[数据增广imgaug库的使用](https://www.cnblogs.com/xxmmqg/p/13062556.html)

[深度学习之数据增强库imgaug使用方法](https://blog.csdn.net/zong596568821xp/article/details/83105700)

[imgaug学习笔记](https://blog.csdn.net/u012897374/article/details/80142744)

