# CNN Image Classification on CIFAR-10

This project implements a Convolutional Neural Network (CNN) model using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The model is trained with data augmentation, batch normalization, dropout, and early stopping to achieve optimal performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Project Overview
The goal of this project is to build and evaluate a CNN model that can classify images from the CIFAR-10 dataset into one of 10 categories. The CIFAR-10 dataset includes images from 10 classes such as airplanes, cars, and birds.

## Dataset
We use the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is split into 50,000 training images and 10,000 testing images.

**Classes:** `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

## Model Architecture
The CNN model consists of the following layers:
- **Convolutional Layers** with ReLU activation and Batch Normalization.
- **MaxPooling Layers** to reduce dimensionality.
- **Dropout Layers** to prevent overfitting.
- **Dense Layers** for final classification, ending with a softmax layer.

## Training
- **Data Augmentation:** Images are augmented with rotation, width/height shifts, and horizontal flips.
- **Optimizer:** Adam with a learning rate of 0.0005.
- **Callbacks:** EarlyStopping and ReduceLROnPlateau are used to optimize training and prevent overfitting.

To train the model:
```python
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    epochs=50,
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

## Evaluation
The model is evaluated on the test dataset after training:
```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
```

## Results
After training, the model achieves a test accuracy of around **76.66%**.

## Usage
1. Clone this repository.
2. Ensure all dependencies are installed (see [Dependencies](#dependencies)).
3. Run the code to train and evaluate the model.

## Dependencies
- TensorFlow
- Matplotlib

Install dependencies with:
```bash
pip install tensorflow matplotlib
```
