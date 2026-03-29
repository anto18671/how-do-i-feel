# Import standard libraries
import json
from pathlib import Path

# Import data libraries
import numpy as np
import pandas as pd

# Import PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

# Import Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# Import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import progress bar
from tqdm import tqdm


# Configuration
SEED = 42
TRAIN_PATH = "data/twitter_training.csv"
VALIDATION_PATH = "data/twitter_validation.csv"
OUTPUT_DIR = "models/minilm-sentiment"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 128


# Define the dataset
class TwitterSentimentDataset(Dataset):
    # Initialize the dataset
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Return dataset length
    def __len__(self):
        return len(self.texts)

    # Return one encoded sample
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Set random seeds
def set_seed():
    # Set PyTorch seed
    torch.manual_seed(SEED)

    # Set NumPy seed
    np.random.seed(SEED)


# Create the output directory
def create_output_directory():
    # Create the output path
    output_path = Path(OUTPUT_DIR)

    # Create the directory if needed
    output_path.mkdir(parents=True, exist_ok=True)


# Get the training device
def get_device():
    # Select CUDA when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print selected device
    print(f"Using device: {device}")

    return device


# Load and prepare the data
def load_data():
    # Load the training CSV
    train_dataframe = pd.read_csv(
        TRAIN_PATH,
        header=None,
        names=["topic_id", "topic", "sentiment", "text"],
    )

    # Load the validation CSV
    validation_dataframe = pd.read_csv(
        VALIDATION_PATH,
        header=None,
        names=["topic_id", "topic", "sentiment", "text"],
    )

    # Extract sorted labels from training data
    unique_labels = sorted(train_dataframe["sentiment"].unique())

    # Build label to index mapping
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    # Build index to label mapping
    index_to_label = {index: label for label, index in label_to_index.items()}

    # Print label mapping
    print(f"Label mapping: {label_to_index}")

    # Convert labels to indices
    train_labels = train_dataframe["sentiment"].map(label_to_index).values
    validation_labels = validation_dataframe["sentiment"].map(label_to_index).values

    # Extract texts
    train_texts = train_dataframe["text"].values
    validation_texts = validation_dataframe["text"].values

    return (
        train_texts,
        train_labels,
        validation_texts,
        validation_labels,
        label_to_index,
        index_to_label,
    )


# Load tokenizer and model
def load_tokenizer_and_model(num_labels, device):
    # Print model loading message
    print("Loading MiniLM-L6 model...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    # Move model to device
    model.to(device)

    return tokenizer, model


# Build dataloaders
def build_dataloaders(train_texts, train_labels, validation_texts, validation_labels, tokenizer):
    # Build training dataset
    train_dataset = TwitterSentimentDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    # Build validation dataset
    validation_dataset = TwitterSentimentDataset(
        texts=validation_texts,
        labels=validation_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    # Build training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # Build validation dataloader
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    return train_loader, validation_loader


# Build optimizer and scheduler
def build_optimizer_and_scheduler(model, train_loader):
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Compute total training steps
    total_steps = len(train_loader) * EPOCHS

    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    return optimizer, scheduler


# Train the model for one epoch
def train_epoch(model, dataloader, optimizer, scheduler, device):
    # Set model to training mode
    model.train()

    # Initialize total loss
    total_loss = 0.0

    # Iterate over batches
    for batch in tqdm(dataloader, desc="Training"):
        # Reset gradients
        optimizer.zero_grad()

        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Run forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Extract loss
        loss = outputs.loss

        # Run backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update optimizer
        optimizer.step()

        # Update scheduler
        scheduler.step()

        # Accumulate loss
        total_loss += loss.item()

    # Return average loss
    return total_loss / len(dataloader)


# Evaluate the model
def evaluate(model, dataloader, device):
    # Set model to evaluation mode
    model.eval()

    # Initialize containers
    all_predictions = []
    all_labels = []
    total_loss = 0.0

    # Disable gradients
    with torch.no_grad():
        # Iterate over validation batches
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Run forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Extract outputs
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate loss
            total_loss += loss.item()

            # Store predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)

            # Store labels
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    # Compute weighted precision, recall, and F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    # Compute average validation loss
    average_loss = total_loss / len(dataloader)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": average_loss,
    }


# Save label mapping
def save_label_mapping(label_to_index, index_to_label):
    # Build output path
    output_path = Path(OUTPUT_DIR) / "label_mapping.json"

    # Build serializable mapping
    label_mapping = {
        "label_to_idx": label_to_index,
        "idx_to_label": {str(key): value for key, value in index_to_label.items()},
    }

    # Save JSON file
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(label_mapping, file, indent=4)


# Save training history
def save_training_history(history):
    # Build output path
    output_path = Path(OUTPUT_DIR) / "training_history.json"

    # Save JSON file
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


# Save the best model and tokenizer
def save_best_model(model, tokenizer):
    # Save model
    model.save_pretrained(OUTPUT_DIR)

    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)


# Run training
def run_training(model, tokenizer, train_loader, validation_loader, optimizer, scheduler, device, label_to_index, index_to_label):
    # Initialize best F1
    best_f1 = 0.0

    # Initialize training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    # Print training header
    print(f"\nStarting training for {EPOCHS} epochs...")

    # Run epoch loop
    for epoch_index in range(EPOCHS):
        # Print epoch header
        print(f"\nEpoch {epoch_index + 1}/{EPOCHS}")

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        history["train_loss"].append(train_loss)

        # Print training loss
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate on validation set
        validation_metrics = evaluate(model, validation_loader, device)

        # Store validation metrics
        history["val_loss"].append(validation_metrics["loss"])
        history["val_accuracy"].append(validation_metrics["accuracy"])
        history["val_f1"].append(validation_metrics["f1"])

        # Print validation metrics
        print(f"Val Loss: {validation_metrics['loss']:.4f}")
        print(f"Val Accuracy: {validation_metrics['accuracy']:.4f}")
        print(f"Val Precision: {validation_metrics['precision']:.4f}")
        print(f"Val Recall: {validation_metrics['recall']:.4f}")
        print(f"Val F1: {validation_metrics['f1']:.4f}")

        # Save the best checkpoint
        if validation_metrics["f1"] > best_f1:
            # Update best F1
            best_f1 = validation_metrics["f1"]

            # Print best model message
            print(f"New best F1 score: {best_f1:.4f}. Saving model...")

            # Save model and tokenizer
            save_best_model(model, tokenizer)

            # Save label mapping
            save_label_mapping(label_to_index, index_to_label)

    # Save training history
    save_training_history(history)

    # Print final result
    print(f"\nTraining complete! Best F1 score: {best_f1:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")


# Define the main function
def main():
    # Set random seeds
    set_seed()

    # Create output directory
    create_output_directory()

    # Get device
    device = get_device()

    # Load data
    print("Loading data...")
    (
        train_texts,
        train_labels,
        validation_texts,
        validation_labels,
        label_to_index,
        index_to_label,
    ) = load_data()

    # Print dataset summary
    num_labels = len(label_to_index)
    print(f"Number of classes: {num_labels}")
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(validation_texts)}")

    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(num_labels, device)

    # Build dataloaders
    train_loader, validation_loader = build_dataloaders(
        train_texts,
        train_labels,
        validation_texts,
        validation_labels,
        tokenizer,
    )

    # Build optimizer and scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader)

    # Run training
    run_training(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
    )


# Run the script
if __name__ == "__main__":
    main()