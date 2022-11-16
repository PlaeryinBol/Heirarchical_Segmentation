# Hierarchical semantic segmentation on the Pascal-part dataset.

Some results:
<p align="center">
    <img src="imgs/examples.jpg", width="700px">
</p>

### Data

[[Google Drive dataset](https://drive.google.com/file/d/1llsnhnfHIypQDmGNB9vl5Yvwjir-zZwU/view?usp=sharing)]
[[Original dataset](http://roozbehm.info/pascal-parts/pascal-parts.html)]

The `JPEGImages` folder contains source images in jpeg format. The `gt_masks` folder contains segmentation masks in numpy format.
There are 7 classes in the dataset, which have the following hierarchical structure (the class index is indicated in brackets):

```
├── (0) background
└── body
    ├── upper_body
    |   ├── (1) low_hand
    |   ├── (6) up_hand
    |   ├── (2) torso
    |   └── (4) head
    └── lower_body
        ├── (3) low_leg
        └── (5) up_leg
```
For each nesting level, it is proposed to calculate the metric separately.  
Main metric: mean Intersection over Union (mIoU).  
Additional metric: Pixel Accuracy.

Finally, the metric values of the resulting model are evaluated in 3 body-level categories:
* Top level mIoU<sup>0</sup> - `body`
* Middle level mIou<sup>1</sup> - `upper_body`, `lower_body`
* Bottom level mIoU<sup>2</sup> - `low_hand`, `up_hand`, `torso`, `head`, `low_leg`, `up_leg`

### Getting started

Download Pascal-part dataset, save the train, val and test data to the appropriate folders - еach folder should contain original jpg-images and their corresponding png-masks.

### Train

Training pipeline presented in a <a href="https://github.com/PlaeryinBol/Heirarchical_segmentation/blob/main/Demo.ipynb">`Demo.ipynb`</a>

### Experiments

|   Method  |    Backbone     |  Img_size | Epochs | Top mIoU | Middle mIoU | Bottom mIoU  | mIoU |
| :-------: | :-------------: | :------: | :----: | :----: | :----: | :----: | :----: | :----: |
|  Unet  |     mobilenet_v2  | 256x256 | 24 | 0.342 | 0.380 | 0.238 | 0.446  |
|  Unet  |     mobilenet_v2  | 512x512 | 24 | 0.390 | 0.417 | 0.286 | 0.490  |
|  Unet  |     resnext101_32x8d  | 512x512 | 17 | 0.326 | 0.364 | 0.194 | 0.433  |
|  Unet  |     vgg19  | 512x512 | 23 | 0.347 | 0.382 | 0.216 | 0.450  |
|  Unet  |     timm-efficientnet-b3  | 512x512 | 24 | 0.348 | 0.315 | 0.216 | 0.423  |
|  DeepLabV3  |     mobilenet_v2  | 512x512 | 29 | 0.393 | 0.422 | 0.286 | 0.492  |
|  FPN  |     mobilenet_v2  | 512x512 | 21 | 0.376 | 0.405 | 0.263 | 0.478  |

Various sets of data augmentations also have been tested.  
Best method: DeepLabV3 with mobilenet_v2 backbone, Top mIoU = 0.393, Middle mIoU = 0.422, Bottom mIoU = 0.286, [[weights](https://drive.google.com/file/d/1DXsWpWbTueSY6f_zobAqoyX5flp15lzV/view?usp=share_link)].

### Ideas for improving results
1. 
2. Оформление результатов.
3. Структура репозитория.
4. Соответствие решения тестовому заданию.
5. Любые релевантные теме мысли, идеи и соображения.