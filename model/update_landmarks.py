# update_landmarks.py
import pandas as pd

# Paths
input_path = r"D:\Study\Project\SignLanguageRecognition\data\landmarks.csv"
output_path = r"D:\Study\Project\SignLanguageRecognition\data\landmarks_updated.csv"

# Load data
try:
    data = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: {input_path} not found.")
    exit()

# Remove space and delete
data = data[~data['label'].isin(['space', 'delete'])]

# Rename columns to match original_landmarks.csv
new_columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
if len(data.columns) != len(new_columns):
    print(f"Error: Column length mismatch. Expected {len(new_columns)}, got {len(data.columns)}")
    exit()
data.columns = new_columns

# Verify
print(f"Rows after filtering: {len(data)}")
print("\nColumns:", list(data.columns))
print("\nLabel counts:\n", data['label'].value_counts())
print("\nNumber of NaN values per column:\n", data.isna().sum())

# Save
data.to_csv(output_path, index=False)
print(f"Saved updated data to {output_path}")