# Wild Animal Image Classification using CNN

This project applies a Convolutional Neural Network (CNN) to classify wildlife images into four categories: buffalo, elephant, rhino, and zebra.

## Overview
The project follows a full machine learning workflow, including dataset filtering, preprocessing, train/validation/test splitting, model training, and evaluation.

## Dataset
The dataset contains wildlife images from four classes:
- buffalo
- elephant
- rhino
- zebra

Source:
[African Wildlife Dataset on Kaggle](https://www.kaggle.com/datasets/biancaferreira/african-wildlife)

## Preprocessing
- Filtered image files from the original dataset
- Split the dataset into:
  - 70% training
  - 15% validation
  - 15% testing
- Resized all images to 128 × 128

## Model
The CNN architecture includes:
- 3 convolutional layers
- max pooling layers
- flatten layer
- dense layer with ReLU
- dropout layer
- softmax output layer for 4-class classification

## Training Configuration
- Framework: TensorFlow / Keras
- Batch size: 32
- Epochs: 15
- EarlyStopping and ModelCheckpoint were used during training

## Results
Final test results:
- Test accuracy: 72.8%
- Test loss: 0.7127

### Classification Report
- Buffalo: Precision 0.83, Recall 0.51, F1-score 0.63
- Elephant: Precision 0.62, Recall 0.61, F1-score 0.62
- Rhino: Precision 0.63, Recall 0.86, F1-score 0.73
- Zebra: Precision 0.90, Recall 0.93, F1-score 0.91

## Files
- `wildlife_cnn_classifier.py` — main training and evaluation code
- `sample-results.txt` — sample output of test results

## Notes
- This project focuses on image classification, not object detection
- The output currently shows final evaluation results printed from the training script
