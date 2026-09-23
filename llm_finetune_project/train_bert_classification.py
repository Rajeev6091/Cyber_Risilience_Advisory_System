# train_bert_classification_with_cm.py
"""
Train a BERT-mini classifier on the provided CSV, optionally apply LoRA (PEFT),
and print/save classification report + confusion matrix (image).

Expected CSV columns:
- 'input' : text to classify
- 'label' : one of "bad", "good", "excellent"

Adjust CSV_PATH, MODEL_NAME, OUTPUT_DIR as needed.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import Dataset
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# Optional: PEFT imports (only needed if you want LoRA)
USE_PEFT = True  # set to False to skip PEFT
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    peft_available = True
except Exception as e:
    print("PEFT not available or import failed. Continuing without LoRA. Error:", e)
    peft_available = False
    USE_PEFT = False

# ======== Config / params ========
# Project directory = folder containing this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset", "csv_dataset", "testing20.csv")
MODEL_NAME = "prajjwal1/bert-mini"
OUTPUT_DIR = os.path.join(BASE_DIR, "bert-mini-finetuned")
MAX_LENGTH = 256
RANDOM_STATE = 42
NUM_LABELS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Label mapping
label2id = {"bad": 0, "good": 1, "excellent": 2}
id2label = {v: k for k, v in label2id.items()}

# ======== Load & validate CSV ========
df = pd.read_csv(CSV_PATH)
if "label" not in df.columns:
    raise ValueError(f"CSV must contain a 'label' column. Path: {CSV_PATH}")
if "input" not in df.columns:
    raise ValueError(f"CSV must contain an 'input' column with the text to classify (column name: 'input'). Path: {CSV_PATH}")

# Map labels to ints
df["label"] = df["label"].map(label2id)

# Sanity check for unmapped labels
if df["label"].isnull().any():
    print("Error: Some labels could not be mapped to integers. Unmapped rows:")
    print(df[df["label"].isnull()])
    raise SystemExit("Please fix label values so they map to 'bad','good','excellent'.")

# ======== Tokenizer & model ========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Resize token embeddings if tokenizer changed
if tokenizer.vocab_size != model.get_input_embeddings().weight.size(0):
    model.resize_token_embeddings(len(tokenizer))

# ======== Optional: Wrap with LoRA (PEFT) ========
if USE_PEFT and peft_available:
    try:
        # For encoder models, "dense" is a broad target module; adjust if you inspect model layers.
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["dense"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        print("Model wrapped with LoRA (PEFT).")
    except Exception as e:
        print("Warning: Failed to apply LoRA. Continuing without PEFT. Error:", e)
        USE_PEFT = False

# ======== Dataset class ========
class BERTClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length=MAX_LENGTH):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["input"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
        )

        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }
        return item

# ======== Train / Eval split (internal) ========
train_df, eval_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=RANDOM_STATE,
)

train_dataset = BERTClassificationDataset(train_df, tokenizer, max_length=MAX_LENGTH)
eval_dataset = BERTClassificationDataset(eval_df, tokenizer, max_length=MAX_LENGTH)

# ======== TrainingArguments ========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=12,
    learning_rate=3e-4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=3,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=1,
)

# ======== Data collator ========
data_collator = DataCollatorWithPadding(tokenizer)

# ======== Metrics func (Trainer uses for eval) ========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ======== Trainer ========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ======== Train ========
trainer.train()

# ======== Save model & tokenizer ========
# If using PEFT, save adapter only; otherwise save full model
try:
    if USE_PEFT and peft_available:
        adapter_out = os.path.join(OUTPUT_DIR, "peft_adapter")
        os.makedirs(adapter_out, exist_ok=True)
        # If model is a PeftModel, save_pretrained will save adapter weights
        model.save_pretrained(adapter_out)
        print(f"Saved PEFT adapter to: {adapter_out}")
        tokenizer.save_pretrained(OUTPUT_DIR)
    else:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Saved full model & tokenizer to: {OUTPUT_DIR}")
except Exception as e:
    print("Warning: saving with PEFT failed or caused error. Falling back to saving full model. Error:", e)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

# ======== Predict on eval set using trainer.predict (recommended) ========
pred_out = trainer.predict(eval_dataset)
logits = pred_out.predictions
if logits is None:
    raise RuntimeError("No predictions returned from trainer.predict()")

preds = np.argmax(logits, axis=-1)
true_labels = pred_out.label_ids

# ======== Print metrics (again) ========
print("\n==== Evaluation metrics (from predictions) ====")
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted", zero_division=0)
acc = accuracy_score(true_labels, preds)
print(f"accuracy: {acc:.4f}  precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}")

# ======== Classification report ========
print("\n========== Classification Report ==========\n")
print(classification_report(true_labels, preds, target_names=["bad", "good", "excellent"], zero_division=0))

# ======== Confusion matrix (raw) ========
cm = confusion_matrix(true_labels, preds)
print("\n========== Confusion Matrix (raw counts) ==========\n")
print(cm)

# ======== Save confusion matrix as PNG (heatmap) ========
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["bad", "good", "excellent"],
                yticklabels=["bad", "good", "excellent"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to: {cm_path}")

    # Also save normalized confusion matrix (percent)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.nan_to_num(cm_norm), annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["bad", "good", "excellent"],
                yticklabels=["bad", "good", "excellent"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Normalized by True Count)")
    cm_norm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png")
    plt.tight_layout()
    plt.savefig(cm_norm_path, dpi=300)
    plt.close()
    print(f"Saved normalized confusion matrix to: {cm_norm_path}")

except Exception as e:
    print("Could not plot/save confusion matrix images (matplotlib or seaborn might be missing). Error:", e)
    print("You can still view the raw confusion matrix above.")

