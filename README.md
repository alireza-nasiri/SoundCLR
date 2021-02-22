# Contrastive-Learning-in-ESC
<h3> Three Training Schemes For The Audio Classifiers: </h3>


<p align="center"><img src="resources/overview.jpg" alt="overview of our three models" height="900"></p>

<h3> Data Augmentation: </h3>
<img src="resources/data_augmentation.jpg" alt="data augmentation process">

### Setup and Dependencies:
First, install the above dependencies.

Second, download ESC50 and US8K datasets and put them inside the 'data' directory

###Quickstart
To train a classifier with ResNet-50 with cross-entropy loss:
```
$ python train_crossEntropyLoss.py
```
To train a classifier with ResNet-50 with supervised-contrastive loss:
```
$ python train_contrastiveLoss.py
```
To train a classifier with ResNet-50 with hybrid loss:
```
$ python train_hybridLoss.py
```
