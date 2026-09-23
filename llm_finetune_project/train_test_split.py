import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Project directory = folder containing this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "csv_dataset")

# Load the dataset
df = pd.read_csv(os.path.join(DATASET_DIR, "DataSet_Cyber.csv"))

# Split the data (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2)

# Save to CSV files
train_df.to_csv(os.path.join(DATASET_DIR, "classified_dataset_train.csv"), index=False)
test_df.to_csv(os.path.join(DATASET_DIR, "classified_dataset_test.csv"), index=False)

print("✅ Dataset split complete.")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
