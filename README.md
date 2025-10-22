# AdVAR-DNN-LCN2025

This repository contains the implementation for our paper  
**"AdVAR-DNN: Adversarial Variational Representation Attack on Distributed Neural Networks"**,  
accepted at **IEEE LCN 2025**.

The project investigates how adversarially modified intermediate features can affect  
distributed neural network inference. It includes scripts for feature extraction,  
VAE training, and generation of manipulated (adversarial) representations.

------------------------
## 🚀 How to Run the Code

### 1️⃣ Install dependencies
Make sure you have Python 3.9 or newer and install the required libraries:

```bash
pip install -r requirements.txt
```
### 2️⃣ Extract intermediate features

Before running Step 2, please download the **retrained VGG19 model on CIFAR-100** used in this project.  
The model was obtained via **transfer learning**, where a pretrained ImageNet VGG19 was fine-tuned on the CIFAR-100 dataset.  
It serves as the **feature extractor** in our experiments, providing intermediate representations  
that are later used for training the Variational Autoencoder (VAE).

🔗 [Download vgg19_cifar100_retrained.h5 from Google Drive]([https://drive.google.com/your-link-here](https://drive.google.com/file/d/1mGUZRYSpoXHNf--rVchxyUVKV5E67s4Z/view?usp=drive_link)

After downloading, place the file in the following directory within the repository:

```bash
python src/edge_cnsm2025/extract_intermediate_features.py
```

