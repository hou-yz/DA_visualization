# Visualizing Adapted Knowledge in Domain Transfer

```
@inproceedings{hou2021visualizing,
  title={Visualizing Adapted Knowledge in Domain Transfer},
  author={Hou, Yunzhong and Zheng, Liang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Under construction


## Overview

This repo dedicates to visualize the learned knowledge in domain adaptation. 
To understand the adaptation process, we portray the knowledge difference between the source and target model with image translation, using the source-free image translation (SFIT) method proposed in our [CVPR2021](http://cvpr2021.thecvf.com/) paper *[Visualizing Adapted Knowledge in Domain Transfer]()*.

Specifically, we feed the generated source-style image to the source model, and the original target image to the target model, formulating two branches respectively. 
Through update the generated image, we force similar outputs between the two branches. When such requirements are met, the image difference should compensate for and can represent the knowledge difference between models. 

## Content
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Run the Code](#run-the-code)
    * [Train source and target models](#train-source-and-target-models)
    * [Visualization](#visualization)


## Dependencies
This code uses the following libraries
- python 3.7+
- pytorch 1.6+ & torchvision
- numpy
- matplotlib
- pillow
- scikit-learn

## Data Preparation
By default, all datasets are in `~/Data/`. We use digits (automatically downloaded), [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/), and [VisDA](http://ai.bu.edu/visda-2017/) datasets. 

Your `~/Data/` folder should look like this
```
Data
├── digits/
│   └── ...
├── office31/ 
│   └── ...
└── visda/
    └── ...
```

## Run the Code

### Train source and target models
Once the data preparation is finished, you can train source and target models using unsupervised domain adaptation (UDA) methods
```shell script
python train_DA.py -d digits --source svhn --target mnist
``` 
Currently, we support [MMD]() ```--da_setting mmd```, [ADDA]() ```--da_setting adda```, and [SHOT]() ```--da_setting shot```.

### Visualization
Based on the trained source and target models, we visualize their knowledge difference via SFIT
```shell script
python train_SFIT.py -d digits --source svhn --target mnist
``` 
