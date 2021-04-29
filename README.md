# Deep_Learning_Data_Augmentation
 深度学习数据增广方法

## 方案一：使用工具imgaug增强数据

### 1、imgaug的介绍

​		imgaug是用于机器学习实验中图像增强的库。它支持广泛的扩充技术，可以轻松地组合它们并以随机顺序或在多个CPU内核上执行它们，具有简单而强大的随机界面，不仅可以扩充图像，还可以扩充关键点/地标，边界框，热图和分段图。

![大量扩充](https://imgaug.readthedocs.io/en/latest/_images/heavy.jpg)

### 2、imgaug的安装与卸载

​		该库使用必须安装的python。支持Python 2.7、3.4、3.5、3.6、3.7和3.8。

2.1、在Anaconda中安装

```python
conda config --add channels conda-forge
conda install imgaug
```

使用扩充器`imgaug.augmenters.imgcorruptlike`需要安装`imagecorruptions`

```python
pip install imagecorruptions
```

2.2、在`pip`中安装

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

2.3、卸载

在Anaconda中卸载

```
conda remove imgaug
```

在pip中卸载

```
pip uninstall imgaug
```



## 方案二：使用工具opencv增强数据