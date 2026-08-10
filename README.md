# 🟡 Neonatal Jaundice Detection Using Deep Learning

A deep learning-based image classification system for detecting neonatal
jaundice from infant skin images using MATLAB.

The project implements both a **Custom Convolutional Neural Network (CNN)**
and **ResNet-18 Transfer Learning** to classify infant skin images into
**Jaundice** and **Normal** categories.

---

## 📌 Project Overview

Neonatal jaundice is a common condition in newborns caused by increased
bilirubin levels. Severe untreated cases can lead to complications such as
kernicterus.

This project explores a non-invasive image-based screening approach using
smartphone-captured infant skin images. The system uses deep learning to
identify visual patterns associated with jaundice.

The main objective is to develop an affordable and accessible screening
approach that can potentially be deployed on resource-constrained devices.

---

## 🎯 Objectives

- Develop a deep learning model for neonatal jaundice classification.
- Classify infant skin images into **Jaundice** and **Normal**.
- Handle class imbalance using class weighting.
- Improve model robustness using image augmentation.
- Compare a Custom CNN with ResNet-18 transfer learning.
- Evaluate the models using accuracy, precision, recall, F1-score and ROC-AUC.
- Implement single-image prediction with confidence scores.

---

## 📊 Dataset

The project uses the **Jaundice Image Dataset** available on Kaggle.

**Dataset:**  
https://www.kaggle.com/datasets/aiolapo/jaundice-image-data

The dataset contains **760 color images**, approximately:

- 75% Normal
- 25% Jaundice

Due to class imbalance, class weighting was applied during model training
to give greater importance to jaundice samples.

> The dataset images are not included in this repository.

---

## 🧠 Models Implemented

### 1. Custom CNN

A lightweight CNN architecture was developed using MATLAB.

Architecture:

```text
Input Image (224 × 224 × 3)
        ↓
Convolution (32 Filters)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Max Pooling
        ↓
Convolution (64 Filters)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Max Pooling
        ↓
Convolution (128 Filters)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Max Pooling
        ↓
Flatten
        ↓
Fully Connected (256)
        ↓
ReLU
        ↓
Dropout (0.5)
        ↓
Fully Connected
        ↓
Softmax
        ↓
Jaundice / Normal
