# ============================================================
# train.py — Full Training Script
# MLOps Assignment 2 | IIT Jodhpur
# ============================================================

import os
import json
import random
import wandb
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset

# ── Device ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Config ──────────────────────────────────────────────────
model_name   = 'distilbert-base-cased'
max_length   = 512
WANDB_KEY    = os.environ.get('WANDB_API_KEY', '')
HF_TOKEN     = os.environ.get('HF_TOKEN', '')
HF_USERNAME  = "your-username"   # ← replace with your HF username
REPO_NAME    = "distilbert-goodreads-genres"

# ── W&B init ────────────────────────────────────────────────
wandb.init(
    project="mlops-assignment2",
    name="distilbert-run-1",
    config={
        "model": model_name,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 3e-5,
        "max_length": max_length,
        "dataset": "UCSD Goodreads",
        "platform": "Kaggle",
    }
)

# ── Tokenizer & Model ────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# ── Dataset Class ────────────────────────────────────────────
class ReviewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# ── Metrics ──────────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted")
    }

# ── Training Arguments ───────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="distilbert-run-1",
)

print("Training script loaded successfully!")
print("Run this inside Kaggle Notebook for GPU access.")