# Model Training

## Overview

The training stage teaches the CNN model to recognize different steel surface categories. During training, the model learns from labeled images, evaluates its performance using the validation dataset, and saves the best-performing model for future predictions.

---

# TRAIN-1: Setup Training

## Purpose

This function sets up the loss function and optimizer used during training.

## What It Does

- Creates the CrossEntropyLoss function.
- Creates the Adam optimizer.
- Sets the learning rate.

## Why Is It Important?

The loss function measures the prediction error, while the optimizer updates the model weights to improve its performance during training.

## TRAIN-1 Code

def setup_training(
    model: nn.Module,
    learning_rate: float = 1e-3,
) -> tuple[nn.Module, optim.Optimizer]:
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return criterion, optimizer




# TRAIN-2: Train One Epoch

## Purpose

This function trains the CNN model for one complete epoch using the training dataset.

## What It Does

- Loads a batch of training images.
- Makes predictions.
- Calculates the training loss.
- Updates the model weights.
- Calculates the training accuracy.

## Why Is It Important?

This is the main learning step where the CNN improves its ability to classify steel surface images.

## TRAIN-2 Code
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:

model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    average_loss = running_loss / len(loader)
    accuracy = correct / total

    return average_loss, accuracy

# TRAIN-3: Validate the Model

## Purpose

This function evaluates the trained model using the validation dataset.

## What It Does

- Loads validation images.
- Makes predictions.
- Calculates the validation loss.
- Calculates the validation accuracy.

## Why Is It Important?

Validation measures how well the model performs on unseen images without updating its weights.

## TRAIN-3 Code

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
   
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    average_loss = running_loss / len(loader)
    accuracy = correct / total

    return average_loss, accuracy

# TRAIN-4: Save the Best Model

## Purpose

This function saves the model whenever the validation accuracy improves.

## What It Does

- Compares the current validation accuracy with the best recorded accuracy.
- Saves the model if the performance has improved.

## Why Is It Important?

Saving the best-performing model ensures that the most accurate version is available for future predictions.

## TRAIN-4 Code

if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "num_classes": NUM_CLASSES,
    }

            torch.save(checkpoint, CHECKPOINT_PATH)

            logger.info("Saved best model to %s", CHECKPOINT_PATH)
    elapsed = time.time() - start_time
    logger.info("Training complete | time=%.1fs | best_val_acc=%.3f", elapsed, best_val_acc)
    return str(CHECKPOINT_PATH)

# Training Configuration

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Image Size: 256 × 256 pixels
- Number of Classes: 5




# Training Results

Include your final training results here after running the model.

Example metrics to report:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Best Validation Accuracy
- Number of Epochs


# Summary
The training stage teaches the CNN model to classify steel surface images by learning from labeled data. The model is evaluated after each epoch, and the best-performing version is saved for use during inference.