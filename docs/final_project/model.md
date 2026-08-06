# CNN Model Architecture

## Overview

The CNN (Convolutional Neural Network) is the core of this project. It learns important features from steel surface images and classifies them into one of five categories.

The model uses convolutional layers to detect image patterns, pooling layers to reduce image size, and fully connected layers to produce the final prediction.

---

# MODEL-1: CNN Architecture

## Purpose

This section defines the structure of the Convolutional Neural Network (CNN). It specifies the layers used to extract image features and classify steel surface images.

## What It Does

The CNN model:

- Uses convolutional layers to detect important image features.
- Applies activation functions to learn complex patterns.
- Uses pooling layers to reduce image size and computation.
- Uses fully connected layers to classify images into five classes.

## Why Is It Important?

The CNN architecture allows the model to automatically learn useful features from images without manually designing feature extractors.

## MODEL-1 Code

def __init__(self, num_classes: int = NUM_CLASSES):
    
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
)


# MODEL-2: Forward Pass

## Purpose

This function defines how an input image moves through the CNN model to produce a prediction.

## What It Does

The forward function:

- Receives an input image.
- Passes the image through the convolutional layers.
- Extracts important image features.
- Passes the extracted features through the fully connected layers.
- Produces the final class prediction.

## Why Is It Important?

The forward function controls the flow of data through the CNN and generates the output used for classification.

## MODEL-2 Code
def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


# Summary

The CNN model learns visual patterns from steel surface images and uses these learned features to classify each image into one of the five defect categories. The forward function defines how images pass through the network to produce predictions.