# AdVAR-DNN-LCN2025

This repository contains the implementation for our paper  
**"AdVAR-DNN: Adversarial Variational Representation Attack on Distributed Neural Networks"**,  
accepted at **IEEE LCN 2025**.

The project investigates how adversarially modified intermediate features can affect  
distributed neural network inference. It includes scripts for feature extraction,  
VAE training, and generation of manipulated (adversarial) representations.


## 🚀 How to Run the Code

### 1️⃣ Install dependencies
Make sure you have Python 3.9 or newer and install the required libraries:

```bash
pip install -r requirements.txt

### 2️⃣ Extract intermediate features
python src/edge_cnsm2025/extract_intermediate_features.py


