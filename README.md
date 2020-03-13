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

---

# Model idea
The backbones we are looking forward to explore are:
- Efficient Net
- Inception Net
- ResNet