# Dataset

## Overview

The dataset module is responsible for organizing the steel surface images before they are used by the CNN model. It scans the dataset folders, assigns labels to each image, loads images during training, and splits the dataset into training, validation, and testing sets.


# DATA-1: Build File List

## Purpose

This function scans all class folders, finds valid image files, and assigns the correct label to each image based on its folder name.

## Steps

- Scan each class folder.
- Find all valid image files.
- Assign a class label to every image.
- Store the image paths and labels in a list.

## Why Is It Important?

The CNN cannot train unless every image has a corresponding label. This function creates the dataset that will be used throughout the project.

## DATA-1 Code

def build_file_list(data_dir: Path | str | None = None) -> list[tuple[str, int]]:
   
    file_list = []

    for folder in data_dir.iterdir():
        if folder.is_dir() and folder.name in CLASS_NAMES:
            label = CLASS_NAMES.index(folder.name)

            for image_path in folder.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    file_list.append((str(image_path), label))

    file_list.sort(key=lambda x: x[0])

    logger.info("Found %d images", len(file_list))

    return file_list

# DATA-2: Load Dataset Samples

## Purpose

This function loads an image from the dataset, prepares it for the CNN model, and returns both the processed image and its label.

## Steps

- Read the image from disk.
- Convert the image to RGB format.
- Apply the preprocessing transforms.
- Return the processed image and its label.

## Why Is It Important?

The model trains one image at a time (or one batch at a time). This function makes sure every image is loaded correctly before it is passed to the CNN.

## DATA-2 Code

def __getitem__(self, idx: int):
        
        image_path, label = self.file_list[idx]

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            result = self.transform(image=image)
            image = result["image"]

        return image, label



# DATA-3: Create Dataset Splits

## Purpose

This function divides the dataset into training, validation, and testing sets.

## Steps

- Separate the dataset into three groups.
- Keep the class distribution balanced using stratified sampling.
- Return the train, validation, and test datasets.

## Why Is It Important?

Splitting the dataset allows the model to learn using the training set while using the validation and test sets to measure its performance on unseen data.

## DATA-3 Code

def create_splits(
    file_list: list[tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    
    labels = [label for _, label in file_list]

    test_ratio = 1.0 - train_ratio - val_ratio

    train_val, test = train_test_split(
        file_list,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    train_val_labels = [label for _, label in train_val]

    val_fraction = val_ratio / (train_ratio + val_ratio)

    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=seed,
    )

    return train, val, test


# Summary

The dataset module prepares the images for the deep learning pipeline by:

- Scanning the dataset folders.
- Assigning labels to every image.
- Loading images during training.
- Creating balanced training, validation, and testing datasets.

This organized dataset allows the CNN model to train and evaluate correctly.



