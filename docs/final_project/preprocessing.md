# Data Preprocessing

## Overview

Before the images are given to the CNN model, they need to be prepared in a consistent format.

The preprocessing stage makes sure that all images have the same size and format so the model can learn effectively.

In this project, preprocessing includes resizing images, normalizing pixel values, applying augmentation, and converting images into PyTorch tensors.

---

# PREPROCESS-1: Training Image Transform

## Purpose

The training transform prepares images that are used to train the CNN model.

During training, the model learns patterns from the images. To help the model learn better and avoid overfitting, some image augmentation techniques are applied.

## Steps Performed

The training preprocessing pipeline performs the following steps:

- Resize images to 256 × 256 pixels.
- Apply random horizontal flipping.
- Adjust brightness and contrast randomly.
- Normalize image pixel values.
- Convert images into PyTorch tensors.

## Why Use Augmentation?

Data augmentation creates slightly different versions of training images. This helps the CNN model learn important features instead of memorizing only the original training images.

For example, the model can learn to recognize a defect even if the image has different lighting conditions or orientation.

## PREPROCESS-1 Code


def build_train_transforms() -> A.Compose:
    
    transform = A.Compose([
        A.Resize(
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1]
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        ToTensorV2(),
    ])

    return transform



# PREPROCESS-2: Validation/Test Image Transform

## Purpose

The validation and test transform prepares images that are used to evaluate the trained CNN model.

These images are processed differently from training images because we want to measure the actual performance of the model on unseen data.

## Steps Performed

The validation/test preprocessing pipeline performs the following steps:

- Resize images to 256 × 256 pixels.
- Normalize image pixel values.
- Convert images into PyTorch tensors.

## Why No Augmentation?

Unlike training images, validation and test images are not randomly modified.

Keeping these images unchanged provides a fair evaluation of how well the model performs on new unseen data.


## PREPROCESS-2 Code

def build_val_transforms() -> A.Compose:
    
    transform = A.Compose([
        A.Resize(
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1]
        ),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        ToTensorV2(),
    ])

    return transform


