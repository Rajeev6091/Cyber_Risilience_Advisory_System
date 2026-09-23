# import pandas as pd
# import numpy as np
# from scipy.special import softmax
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# # Label mapping
# label2id = {"bad": 0, "good": 1, "excellent": 2}

# # Load your CSV
# df = pd.read_csv("/Users/Rajeev/Documents/Cyber Resilience Project/llm_finetune_project/dataset/csv_dataset/OriginalData.csv")

# # df= pd.read_csv("/Users/Rajeev/Documents/Cyber Resilience Project/llm_finetune_project/dataset/csv_dataset/DataSet_Cyber.csv")
# # Map labels to integers
# df["label"] = df["label"].map(label2id)

# # new things
# # Sanity check for unmapped labels
# if df["label"].isnull().any():
#     print("Error: Some labels could not be mapped to integers.")
#     print(df[df["label"].isnull()])
#     exit(1)  # Stop the script

# # Load a lightweight BERT tokenizer (very fast on M1)
# model_name = "prajjwal1/bert-mini"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


# # Define max sequence length
# MAX_LENGTH = 256

# # Custom Dataset for BERT
# class BERTClassificationDataset(Dataset):
#     def __init__(self, dataframe, tokenizer):
#         self.data = dataframe
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         inputs = self.tokenizer(
#             row["input"],
#             truncation=True,
#             padding="max_length",
#             max_length=MAX_LENGTH,
#             return_tensors="pt"
#         )

#         return {
#             "input_ids": inputs["input_ids"].squeeze(),
#             "attention_mask": inputs["attention_mask"].squeeze(),
#             "label": int(row["label"])
#         }

# # Create dataset
# # dataset = BERTClassificationDataset(df, tokenizer)

# train_df, eval_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
# train_dataset = BERTClassificationDataset(train_df, tokenizer)
# eval_dataset = BERTClassificationDataset(eval_df, tokenizer)

# # 5. Training Arguments
# training_args = TrainingArguments(
#     output_dir="/Users/Rajeev/Documents/Cyber Resilience Project/llm_finetune_project/bert-mini-finetuned",
#     per_device_train_batch_size=16,
#     num_train_epochs=30,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir="./logs",
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy"
# )

# # 6. Define compute_metrics function
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# # TEMPERATURE = 0.5  # <--- your temperature
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = logits.argmax(axis=-1)            #---------------
    
#     # apply temperature scaling
#     # scaled_logits = logits / TEMPERATURE       #---------------
#     # probs = softmax(scaled_logits, axis=-1)     #---------------
#     # preds = np.argmax(probs, axis=-1)            #---------------

#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     acc = accuracy_score(labels, preds)
#     return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# # 7. Create Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset, # For demo, using train as eval; replace with real eval split!
#     data_collator=DataCollatorWithPadding(tokenizer),
#     compute_metrics=compute_metrics
# )

# # 8. Start training
# trainer.train()

# trainer.save_model("bert-mini-finetuned")
# tokenizer.save_pretrained("bert-mini-finetuned")






import os
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model, TaskType

# Project directory = folder containing this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "csv_dataset")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "bert-mini-finetuned")

# -----------------------------
# Label mapping
# -----------------------------
label2id = {"bad": 0, "good": 1, "excellent": 2}
id2label = {0: "bad", 1: "good", 2: "excellent"}

MAX_LENGTH = 256


# -----------------------------
# Custom Dataset Class
# -----------------------------
class BERTClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        inputs = self.tokenizer(
            row["input"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": int(row["label"])  # IMPORTANT: must be "labels"
        }


# -----------------------------
# Metrics Function
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# -----------------------------
# Training Logic
# -----------------------------
def train_model():

    print("Loading dataset...")

    df = pd.read_csv(os.path.join(DATASET_DIR, "OriginalData.csv"))

    df["label"] = df["label"].map(label2id)

    if df["label"].isnull().any():
        raise ValueError("Some labels could not be mapped properly.")

    model_name = "prajjwal1/bert-mini"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Transformer Fine-Tuning for Sequence Classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    # LoRA Configuration  (LoRA-based Parameter Efficient Fine-Tuning (PEFT).)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,   # sequence classification
        r=8,                          # LoRA rank
        lora_alpha=32,                # scaling factor
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"]  # BERT attention layers
        )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train_dataset = BERTClassificationDataset(train_df, tokenizer)
    eval_dataset = BERTClassificationDataset(eval_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=100,  # reduced from 30 (better)
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print("Training completed successfully!")


# -----------------------------
# Prevent Auto Execution
# -----------------------------
if __name__ == "__main__":
    train_model()