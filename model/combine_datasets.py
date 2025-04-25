# combine_datasets.py
import pandas as pd

# Paths
landmarks_path = r"D:\Study\Project\SignLanguageRecognition\data\landmarks_updated.csv"
original_path = r"D:\Study\Project\SignLanguageRecognition\data\original_landmarks.csv"
output_path = r"D:\Study\Project\SignLanguageRecognition\data\combined_landmarks.csv"

# Load data
try:
    landmarks = pd.read_csv(landmarks_path)
    original = pd.read_csv(original_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Verify columns match
if set(landmarks.columns) != set(original.columns):
    print(f"Error: Columns do not match. Landmarks: {landmarks.columns}, Original: {original.columns}")
    exit()

# Combine
combined = pd.concat([landmarks, original], ignore_index=True)

# Verify labels
expected_signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'delete', 'nothing']
if set(combined['label'].unique()) != set(expected_signs):
    print(f"Error: Expected {expected_signs}, got {combined['label'].unique()}")
    exit()

# Check for NaN values
if combined.isna().sum().sum() > 0:
    print(f"Error: Found {combined.isna().sum().sum()} NaN values.")
    exit()

# Save
combined.to_csv(output_path, index=False)
print(f"Saved {len(combined)} rows to {output_path}")