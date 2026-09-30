# Driver Drowsiness Detection Using MobileNetV2

## Overview

This project implements a deep learning-based driver drowsiness detection system using transfer learning with MobileNetV2.

The model classifies facial images into four alertness-related categories:

* Closed
* Open
* No Yawn
* Yawn

The project uses the Yawn Eye Dataset from Kaggle and demonstrates an end-to-end computer vision workflow including dataset preparation, image preprocessing, data augmentation, transfer learning, model training, performance visualization, confusion matrix analysis, classification reporting, and prediction on new images.

## Dataset

The project uses the **Yawn Eye Dataset** from Kaggle.

Dataset source:

[Yawn Eye Dataset - Kaggle](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new?utm_source=chatgpt.com)

The notebook downloaded the dataset directly using the Kaggle API.

### Dataset Classes

| Class   | Description        |
| ------- | ------------------ |
| Closed  | Eyes closed        |
| Open    | Eyes open          |
| no_yawn | No visible yawning |
| yawn    | Yawning            |

The notebook contains:

* Training images: 2,467
* Test images: 433
* Total images: 2,900
* Number of classes: 4

## Project Objective

The objective is to build a computer vision model capable of identifying visual indicators associated with driver drowsiness, particularly eye closure and yawning.

Such a system can serve as a foundation for driver monitoring and alertness-assistance applications.

## Technology Stack

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn
* Kaggle API
* Google Colab

## Methodology

### 1. Dataset Acquisition

The dataset is downloaded directly from Kaggle using the Kaggle API and extracted into the notebook environment.

### 2. Image Preprocessing

Images are resized to:

```text
128 × 128 × 3
```

Pixel values are normalized using:

```python
rescale=1./255
```

### 3. Data Augmentation

The training pipeline applies:

* Rotation up to 20 degrees
* Zoom augmentation up to 20%
* Horizontal flipping
* Pixel normalization

These transformations increase variation in the training images and help the model learn more robust visual features.

### 4. Transfer Learning

The project uses **MobileNetV2 pretrained on ImageNet** as the convolutional feature extractor.

The pretrained base is frozen:

```python
base_model.trainable = False
```

A custom classification head is added:

```text
MobileNetV2
     |
Global Average Pooling
     |
Dropout (0.5)
     |
Dense Layer
     |
Softmax
     |
4 Classes
```

### 5. Model Configuration

| Parameter                    | Configuration            |
| ---------------------------- | ------------------------ |
| Architecture                 | MobileNetV2              |
| Pretrained weights           | ImageNet                 |
| Input size                   | 128 × 128 × 3            |
| Output classes               | 4                        |
| Dropout                      | 0.5                      |
| Optimizer                    | Adam                     |
| Loss                         | Categorical Crossentropy |
| Batch size                   | 32                       |
| Epochs                       | 8                        |
| Trainable MobileNetV2 layers | 0                        |

The model contains approximately **2.26 million parameters**, all of which are frozen in the notebook's configuration.

## Training Results

The training process showed the following validation accuracy:

| Epoch | Training Accuracy | Validation Accuracy |
| ----: | ----------------: | ------------------: |
|     1 |            58.98% |              82.91% |
|     2 |            80.36% |              83.37% |
|     3 |            82.88% |              86.37% |
|     4 |            84.17% |              86.14% |
|     5 |            86.19% |              85.68% |
|     6 |            84.86% |              89.15% |
|     7 |            86.16% |              86.84% |
|     8 |            86.94% |              88.68% |

The highest recorded validation accuracy during training was **89.15% at Epoch 6**.

The final epoch achieved **88.68% validation accuracy**.

## Model Evaluation

The notebook additionally generates:

* Training and validation accuracy curves
* Training and validation loss curves
* Confusion matrix
* Classification report
* Prediction on a new image

### Classification Report

The notebook's final classification report produced approximately:

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------: | -----: | -------: |
| Closed           |      0.27 |   0.28 |     0.27 |
| Open             |      0.27 |   0.27 |     0.27 |
| no_yawn          |      0.23 |   0.32 |     0.27 |
| yawn             |      0.30 |   0.19 |     0.23 |
| Overall Accuracy |           |        |      26% |

### Evaluation Note

There is a discrepancy between the approximately **89% validation accuracy observed during training** and the **26% accuracy reported by the later classification report**.

The notebook evaluates the test generator after configuring it with image augmentation, while predictions are compared directly against `test_data.classes`. The generator's default shuffling behavior can also cause the prediction order to differ from the class-label order.

Therefore, the 26% classification-report result should not be interpreted as the model's definitive test accuracy without correcting the evaluation pipeline.

A more reliable evaluation would use a non-augmented test generator with:

```python
shuffle=False
```

and then generate predictions against the corresponding labels.

## New Image Prediction

The notebook also tests the trained model using a new image:

```text
yawn.png
```

The model predicted:

```text
yawn
```

## Saved Model

The trained model is saved as:

```text
advanced_drowsiness_model.keras
```

This allows the trained model to be reused for future inference without retraining the entire network.

## Project Structure

```text
Driver-Drowsiness-Detection-MobileNetV2/
│
├── drowsiness_detection.ipynb
├── advanced_drowsiness_model.keras
├── README.md
└── sample_images/
```

## Key Skills Demonstrated

* Computer Vision
* Image Classification
* Deep Learning
* Transfer Learning
* MobileNetV2
* TensorFlow and Keras
* Image Data Augmentation
* Model Evaluation
* Confusion Matrix Analysis
* Classification Metrics
* Model Serialization
* Kaggle API
* Python

## Future Improvements

Potential improvements to make the system more reliable include:

1. Correct the test evaluation pipeline using `shuffle=False`.
2. Remove augmentation from the test dataset.
3. Fine-tune selected MobileNetV2 layers after initial training.
4. Evaluate class imbalance and per-class performance.
5. Add precision, recall and F1-score monitoring during training.
6. Test the model on a separate real-world dataset.
7. Build a real-time webcam-based drowsiness detection system.
8. Add temporal analysis across consecutive video frames.
9. Implement an alert mechanism when prolonged drowsiness is detected.

## Disclaimer

This project is an educational machine learning implementation and should not be considered a safety-certified driver monitoring system. Real-world deployment would require substantially broader testing, validation under different environmental conditions, and appropriate safety engineering.

## Author

**Nayanendhu Unnikrishnan**

GitHub: `nayanendhucu`

LinkedIn: `nayanendhu-unnikrishnan`
