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
- [x] Models
    - [x] Efficientnet B7
    - [x] Efficientnet B4
    - [x] Resnet 34
    - [x] Densenet 161
- [x] Loss
    - [x] Cross Entropy Loss
    - [x] Focal loss
    - [x] ArcFace loss
    - [ ] Mean ROC AUC loss ( Kaggle )
- [x] Optimizer
    - [x] RMS prop with Efficient net params
    - [X] Adam
    - [x] AdamW
- [X] Scheduler
    - [X] CosineAnelingLR
    - [X] StepLR

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
|-- folds
|   `-- fgvc7
|       |-- 0
|       |   |-- train
|       |   `-- val
|       |-- 1
|       |   |-- train
|       |   `-- val
|       |-- 2
|       |   |-- train
|       |   `-- val
|       |-- 3
|       |   |-- train
|       |   `-- val
|       `-- 4
|           |-- train
|           `-- val
|-- losses
|-- models
|-- optimisers
|-- pretrained_weights
|-- results
|-- runs
|-- schedulers
|-- transformer
`-- utils

```

---

## Config structure

### Training
Sample files ( train_dummy_1, train_dummy_2 ) are provided in the examples directory


```
mode: train
seed: (optional)
validation_frequency: ( optional )
epochs:
batch_size: ( optional )
num_classes: 
train_dataset: 
  name:
  fold: ( optional )
  resize_dims:
  transform: (optional, uses default if not specified)
val_dataset: 
  name:
  fold: ( optional )
  resize_dims:
  transform: (optional, uses default if not specified)
model: 
  name:
  pred_type: regression/classification/mixed
  tuning_type: feature-extraction/fine-tuning
  hyper_params: ( depends on the model )
    key_1:  ( depends on the model )
    key_2:  ( depends on the model )
  pre_trained_path: ( optional key to resume training from another experiment )
  weight_type: ( optional key to pick the type of weight - best_val_loss/best_val_roc )
optimiser: 
  name: 
  hyper_params:
    key_1: ( depends on the optimiser )
    key_2: ( depends on the optimiser )
scheduler:
  name: 
  hyper_params:
    key_1: ( depends on the scheduler )
    key_2: ( depends on the scheduler )
loss_function: 
  name: ( in case of mixed, this will be the regression loss )
  hyper_params:
    classification_loss: ( in case of mixed )
    classification_coefficient: ( in case of mixed - (0, 1) )
```

### Test
Sample files ( test_dummy ) are provided in the examples directory

```
mode: test
test_dataset: 
  name:
  resize_dims:
ensemble: True/False
num_classes:
experiment_list:
  - experiment:
      path: ( valid train config file name )
      weight_type: ( optional key to pick the type of weight - best_val_loss/best_val_roc )
      resize_dims: ( optional override - original/integer - Use the training config values or the integer passed )
  - experiment:
      path: ( valid train config file name )
      weight_type: ( optional key to pick the type of weight - best_val_loss/best_val_roc )
```

## How to use
#### 1. Using it for train/test
simply modify the run.sh with the required commands:  
```
python main.py "train_dummy_1.yml"
python main.py "test_dummy.yml"
```  
  
#### 2. Using it for augmentation search
Add ```-a``` flag with the training config that you are going to use for scoring:
```
python main.py -a 'train_dummy_1.yml'
```  
  
#### 3. Publish mode while training
Add ```-p``` flag with the training config to publish results to W&B and Telegram:
```
python main.py -p 'train_dummy_1.yml'
```

## Lib dependency
```bash
pip install --user torch wandb torchvision catalyst albumentations pandas scikit-image tqdm scikit-learn pyyaml blessed pyzmq pretrainedmodels tensorboard transformers
```

## Utility commands
This command will monitor GPU status and keep updating the terminal every 0.5 seconds  
```bash
watch -n0.5 nvidia-smi
```
