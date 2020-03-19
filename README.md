# CVPR 2020 - FGVC ( Plant Pathology )
The objective of this challenge is to train a model from **images** that can:  
- Classify leaves into diseased / healthy class ( *Mutually exclusive* classes - **softmax applicable** )
- A leaf may be either of the 4 classes 
    - **healthy** 
    - **multiple_diseases** 
    - **rust**
    - **scab**
- Should be able to handle **unbalanced classes** ( on class occurring very *rarely* ) and **novel symptoms** ( ??? )
- Should be able to accommodate for **perception changes** ( lighting, angle, etc )
- The model should **learn relevant details** from image ( ??? ) - Possible use of latent vector

---

## Dataset
- [Kaggle Page](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data)
- [Download link](https://www.kaggle.com/c/18648/download-all)
- [Supplementary Data for TL](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/discussion/135065)
    - [Alt kaggle link](https://www.kaggle.com/xhlulu/leafsnap-dataset)

---

## EDA
- https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models
    - ( Amazing "Parallel categories plot of targets", Distribution of red channel values", EfficientNet NoisyStudent Prediction (great images) )
- https://www.kaggle.com/pestipeti/eda-plant-pathology-2020
- [Google Sheet - Submitted approaches](https://docs.google.com/spreadsheets/d/1VVi2HST5m4LFaSr-GBiUgQdsM6oA8O-mfq3UFKyqZU8/edit?usp=sharing)

### Summary
- The data contains a **few duplicates** and has both **portrait** and **landscape** images.
- Label distribution:
    - Scab ~32%
    - Rust ~32%
    - Healthy ~28%
    - Multiple ~5%
- Color channel ( RGB ) analysis shows greater variance in B and G channels a compared to R channel.
- Parallel Category plot has guaranteed us that the labels are indeed mutually exclusive

---

## Model Backbone ideas
The backbones we are looking forward to explore are:
- ResNets
- EfficientNet ( B7 )
- SEResNeXts
- DenseNets ( 121 )
- EfficientNet NoisyStudent
- Ensembling ( if needed )

## Augmentation ideas
- Canny Edge detection ( to crop out only the main leaf image )
- Flipping ( horizontal and vertical )
- Using convolutions to generate sunlight effect ( Possibility for low light images ??? )
- Blurring

## Training ideas
- Train.py framework for quick testing
- Cross fold validation
- Stratified K fold validation

## Pretrained Model Weights ( GDrive links )
- [Efficientnet B7](https://drive.google.com/drive/folders/1yza0tPWpx0t6qh6Pfqq_aZ_FrJ0RqCgc?usp=sharing)

---

## Plans
- [x] Dataloaders 
    - [x] Stratified k fold
- [ ] Models
    - [ ] Efficientnet B7
- [ ] Loss
    - [ ] Focal loss
    - [ ] Mean ROC AUC loss ( Kaggle )
- [ ] Optimizer
    - [ ] RMS prop with Efficient net params

---

## Directory structure
```
.
|-- config
|-- data
|   `-- fgvc7
|       `-- images
|-- dataset
|-- docs
|-- losses
|-- models
|-- optimisers
|-- pretrained_weights
|-- schedulers
`-- transformers
```

---

## Lib dependency
```bash
pip install --user torch torchvision catalyst albumentations pandas scikit-image tqdm scikit-learn pyyaml
```
