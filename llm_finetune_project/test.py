import os
import pandas as pd
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from train_finetune import BERTClassificationDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Project directory = folder containing this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "csv_dataset")

# Load test CSV
label2id = {"bad": 0, "good": 1, "excellent": 2}

df = pd.read_csv(os.path.join(DATASET_DIR, "testing20.csv"))
df["label"] = df["label"].map(label2id)

# Load fine-tuned model and tokenizer
model_path = os.path.join(BASE_DIR, "bert-mini-finetuned")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model_name = "prajjwal1/bert-mini"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create Dataset
test_dataset = BERTClassificationDataset(df, tokenizer)

# Define compute_metrics (optional, for detailed metrics)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Initialize Trainer with eval dataset and metrics
trainer = Trainer(
    model=model,
    data_collator=DataCollatorWithPadding(tokenizer),
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Evaluate on test dataset
metrics = trainer.evaluate()
print(metrics)

# predictions = trainer.predict(test_dataset)
# y_true = predictions.label_ids
# y_pred = np.argmax(predictions.predictions, axis=1)

# cm = confusion_matrix(y_true, y_pred)

# print("\n==== Confusion Matrix ====")
# print(cm)

# print("\n==== Classification Report ====")
# print(classification_report(y_true, y_pred, target_names=["bad", "good", "excellent"]))