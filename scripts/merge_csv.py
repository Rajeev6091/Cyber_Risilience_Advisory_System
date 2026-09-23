import os
import pandas as pd

# Project root = parent of this scripts/ directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "llm_finetune_project", "dataset", "csv_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Load both datasets
original = pd.read_csv(os.path.join(DATASET_DIR, "DataSet_Cyber.csv"))
synthetic = pd.read_csv(os.path.join(DATASET_DIR, "synthetics.csv"))

# Concatenate them
merged = pd.concat([original, synthetic], ignore_index=True)

# Save to new file
os.makedirs(OUTPUT_DIR, exist_ok=True)
merged.to_csv(os.path.join(OUTPUT_DIR, "merged_dataset.csv"), index=False)