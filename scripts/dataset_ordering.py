import os
import pandas as pd

# Project root = parent of this scripts/ directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "llm_finetune_project", "dataset", "csv_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Load the CSV file
df = pd.read_csv(os.path.join(DATASET_DIR, "classified_dataset_train.csv"))

# Clean and normalize labels (optional but recommended)
df['label'] = df['label'].str.strip().str.lower()

# Split into individual label groups
bad_df = df[df['label'] == 'bad'].reset_index(drop=True)
good_df = df[df['label'] == 'good'].reset_index(drop=True)
excellent_df = df[df['label'] == 'excellent'].reset_index(drop=True)

# Check they all have 41 rows
if not (len(bad_df) == len(good_df) == len(excellent_df)):
    print("Warning: Counts of labels do not match!")
    print(f"Bad: {len(bad_df)}, Good: {len(good_df)}, Excellent: {len(excellent_df)}")

# Create the ordered list of rows
ordered_rows = []
for i in range(41):  # Since all have 41
    ordered_rows.append(bad_df.iloc[i])
    ordered_rows.append(good_df.iloc[i])
    ordered_rows.append(excellent_df.iloc[i])

# Convert back to DataFrame
final_df = pd.DataFrame(ordered_rows)

# Save to CSV
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "reordered_output.csv")
final_df.to_csv(output_path, index=False)

print(f"✅ Reordering complete. Saved to '{output_path}'.")
