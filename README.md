# 🟡 Neonatal Jaundice Detection Using Deep Learning

A MATLAB-based deep learning system for detecting neonatal jaundice from infant skin images using image classification.

The project implements two deep learning approaches:

* **Custom Convolutional Neural Network (CNN)**
* **ResNet-18 Transfer Learning**

The models classify infant skin images into two categories:

**Jaundice** or **Normal**

---

## 📌 Project Overview

Neonatal jaundice is a common condition in newborns caused by increased bilirubin levels. Severe untreated cases can lead to serious complications such as kernicterus.

This project explores a non-invasive image-based screening approach using infant skin images captured under controlled conditions. A deep learning model analyzes the image and predicts whether it belongs to the **Jaundice** or **Normal** class.

The system was developed in **MATLAB** using the **Deep Learning Toolbox** and **Image Processing Toolbox**.

---

## 🎯 Objectives

* Develop an image-based neonatal jaundice classification system.
* Implement a lightweight Custom CNN architecture.
* Apply ResNet-18 transfer learning for comparison.
* Handle class imbalance using class weighting.
* Improve model robustness using image augmentation.
* Evaluate model performance using multiple classification metrics.
* Implement single-image prediction with confidence scores.
* Explore the possibility of lightweight deployment on resource-constrained devices.

---

## 📊 Dataset

The project uses the **Jaundice Image Dataset** available on Kaggle.

**Dataset:**
https://www.kaggle.com/datasets/aiolapo/jaundice-image-data

### Dataset Statistics

* **Total Images:** 760
* **Normal:** approximately 75%
* **Jaundice:** approximately 25%

Because the dataset is imbalanced, **class weighting** was applied during training to give greater importance to jaundice samples.

> The original dataset is not included in this repository.

---

# 🧠 Models Implemented

## 1. Custom CNN

A lightweight Custom Convolutional Neural Network was developed specifically for the binary classification task.

### Architecture

```text
Input Image
224 × 224 × 3
      ↓
Convolution Layer
32 Filters
      ↓
Batch Normalization
      ↓
ReLU Activation
      ↓
Max Pooling
      ↓
Convolution Layer
64 Filters
      ↓
Batch Normalization
      ↓
ReLU Activation
      ↓
Max Pooling
      ↓
Convolution Layer
128 Filters
      ↓
Batch Normalization
      ↓
ReLU Activation
      ↓
Max Pooling
      ↓
Flatten
      ↓
Fully Connected Layer
256 Neurons
      ↓
ReLU Activation
      ↓
Dropout
0.5
      ↓
Fully Connected Layer
2 Classes
      ↓
Softmax
      ↓
Jaundice / Normal
```

### Custom CNN Features

* Input size: **224 × 224 × 3**
* Convolution filters: **32 → 64 → 128**
* Batch Normalization
* ReLU activation
* Max Pooling
* Fully Connected Layer: **256 neurons**
* Dropout: **0.5**
* Softmax classification
* Class-weighted training

---

## 2. ResNet-18 Transfer Learning

A pre-trained **ResNet-18** network was used as a second deep learning approach.

The pre-trained network was modified for the neonatal jaundice classification task by replacing its final classification layers.

### Architecture

```text
Input Image
      ↓
Pre-trained ResNet-18
      ↓
Feature Extraction
      ↓
Modified Fully Connected Layer
      ↓
2 Output Classes
      ↓
Weighted Classification Layer
      ↓
Jaundice / Normal
```

### ResNet-18 Modifications

The pre-trained ResNet-18 network was loaded and converted into a layer graph.

The original `fc1000` layer was replaced with a new fully connected layer configured for the two target classes.

The learning-rate factors of the newly added fully connected layer were increased to accelerate learning.

The original classification layer was replaced with a custom weighted classification layer to account for class imbalance.

---

## 3. Model Comparison

The project compares two different approaches:

| Feature             | Custom CNN            | ResNet-18             |
| ------------------- | --------------------- | --------------------- |
| Architecture        | Custom CNN            | Pre-trained ResNet-18 |
| Learning Approach   | Training from scratch | Transfer Learning     |
| Convolution Filters | 32 → 64 → 128         | Pre-trained           |
| Optimizer           | Adam                  | Adam                  |
| Epochs              | 10                    | 10                    |
| Mini-Batch Size     | 16                    | 16                    |
| Learning Rate       | 0.001                 | 0.0001                |
| Class Weighting     | Yes                   | Yes                   |
| Data Augmentation   | Yes                   | Yes                   |
| Classification      | Jaundice / Normal     | Jaundice / Normal     |

The Custom CNN provides a lightweight architecture designed specifically for this task, while ResNet-18 provides a transfer-learning approach using features learned from a pre-trained network.

---

# 🔧 Data Preprocessing

The dataset was prepared using MATLAB's image processing and deep learning tools.

### Preprocessing Steps

* Images resized to **224 × 224 pixels**
* Pixel values normalized
* Training and validation datasets created
* Images automatically labeled using folder names
* Class distribution calculated
* Class weights calculated to handle imbalance

---

# 🔄 Data Augmentation

Training images were augmented in real time to improve model robustness.

The following transformations were applied:

* Horizontal reflection
* Rotation between **−15° and +15°**
* Scaling between **0.8 and 1.2**
* X-axis translation
* Y-axis translation

MATLAB functions used:

```matlab
imageDataAugmenter
augmentedImageDatastore
```

---

# ⚙️ Training Configuration

| Parameter             | Custom CNN | ResNet-18 |
| --------------------- | ---------- | --------- |
| Optimizer             | Adam       | Adam      |
| Maximum Epochs        | 10         | 10        |
| Mini-Batch Size       | 16         | 16        |
| Initial Learning Rate | 0.001      | 0.0001    |
| Class Weighting       | Enabled    | Enabled   |
| Data Augmentation     | Enabled    | Enabled   |
| Classification        | Binary     | Binary    |

---

# 📈 Model Evaluation

The `evaluate_model.m` script evaluates the trained models using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* ROC-AUC

Special attention is given to **jaundice recall**, because the dataset contains fewer jaundice samples and false-negative predictions are particularly important for this screening task.

---

# 🏆 Results

The reported Custom CNN training run achieved:

| Metric                |      Result |
| --------------------- | ----------: |
| Validation Accuracy   |  **73.68%** |
| Epochs Completed      | **10 / 10** |
| Total Iterations      |     **370** |
| Initial Learning Rate |   **0.001** |

The project also investigates improving jaundice detection through class weighting and image augmentation.

> The reported result is from the project training run and should not be interpreted as clinical diagnostic accuracy.

---

# 🔍 Single Image Prediction

The `predict_image.m` script allows a user to test an individual image using a trained model.

It provides:

* Predicted class
* Confidence percentage
* Class probabilities
* Prediction visualization

### Example

```matlab
predict_image('path/to/image.jpg')
```

### Prediction Flow

```text
Input Image
     ↓
Image Resize
     ↓
Trained CNN Model
     ↓
Classification
     ↓
Jaundice / Normal
     ↓
Confidence Score
```

---

# 📁 Project Structure

```text
neonatal-jaundice-detection/
│
├── README.md
├── .gitignore
│
├── train_model.m
├── train_resnet.m
├── evaluate_model.m
└── predict_image.m
```

### MATLAB Files

| File               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `train_model.m`    | Trains the Custom CNN model                                  |
| `train_resnet.m`   | Trains the ResNet-18 transfer-learning model                 |
| `evaluate_model.m` | Calculates evaluation metrics and generates evaluation plots |
| `predict_image.m`  | Performs prediction on a single image                        |

---

# ▶️ How to Run

## 1. Requirements

Install MATLAB with:

* MATLAB
* Deep Learning Toolbox
* Image Processing Toolbox

---

## 2. Download the Dataset

Download the Jaundice Image Dataset from Kaggle:

https://www.kaggle.com/datasets/aiolapo/jaundice-image-data

Organize the dataset as:

```text
train/
├── jaundice/
└── normal/

validation/
├── jaundice/
└── normal/
```

---

## 3. Train the Custom CNN

Open MATLAB and run:

```matlab
train_model
```

The trained model will be saved as:

```text
jaundice_model.mat
```

---

## 4. Train ResNet-18

Run:

```matlab
train_resnet
```

The trained ResNet-18 model will be saved as:

```text
jaundice_resnet_model.mat
```

---

## 5. Evaluate the Model

Run:

```matlab
evaluate_model
```

The evaluation script calculates classification metrics and generates visualizations such as the confusion matrix and ROC curve.

---

## 6. Predict a New Image

Run:

```matlab
predict_image('path/to/image.jpg')
```

The script displays the predicted class, confidence score, and class probabilities.

---

# 🛠️ Technologies Used

* **MATLAB**
* **Deep Learning Toolbox**
* **Image Processing Toolbox**
* **Convolutional Neural Networks**
* **ResNet-18**
* **Transfer Learning**
* **Image Classification**
* **Image Augmentation**

---

# 🚀 Future Improvements

Potential future improvements include:

* ResNet-50 / EfficientNet transfer learning
* Grad-CAM based explainability
* Larger and more diverse datasets
* Improved image-quality and lighting normalization
* Mobile application deployment
* Raspberry Pi / edge-device deployment
* Additional clinical validation

---

# 👨‍💻 Authors

**Mohit Ranjan**
**Mayur Jalan**
**Vivek Kumar**

Department of Electronics and Communication Engineering
**Birla Institute of Technology, Mesra**

### Supervisor

**Dr. Akash Kumar Gupta**
Assistant Professor
Department of Electronics and Communication Engineering
BIT Mesra

---

# 📚 References

1. Jaundice Image Dataset — Kaggle
   https://www.kaggle.com/datasets/aiolapo/jaundice-image-data

2. Diagnosis of Jaundice — Kaggle Notebook

3. Abdulrazzak et al. (2024), *Real-Time Jaundice Detection in Neonates*

4. *Explainable Deep Learning for Neonatal Jaundice Classification* (2025)

5. *EDNJIC-KELM: Vision Transformer + KELM for Neonatal Jaundice* (2025)

6. *Neonatal Jaundice Detection Using CNN Algorithm* (2023)

---

# ⚠️ Disclaimer

This project is an **academic/research prototype** for image-based screening and classification.

It is **not intended to replace clinical diagnosis, bilirubin testing, or professional medical advice.**
