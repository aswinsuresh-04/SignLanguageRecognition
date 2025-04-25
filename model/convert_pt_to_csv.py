# convert_pt_to_csv.py
import torch
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Paths
x_data_path = r"D:\Study\Project\SignLanguageRecognition\torch_data\x_data.pt"
y_data_path = r"D:\Study\Project\SignLanguageRecognition\torch_data\y_data.pt"
label_encoder_path = r"D:\Study\Project\SignLanguageRecognition\torch_data\label_encoder.pt"
output_path = r"D:\Study\Project\SignLanguageRecognition\data\original_landmarks.csv"

# Create data directory if it doesn't exist
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Load data
try:
    x_data = torch.load(x_data_path, weights_only=False)
    y_data = torch.load(y_data_path, weights_only=False)
except Exception as e:
    print(f"Error loading x_data.pt or y_data.pt: {e}")
    exit()

# Try to load label encoder
try:
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading label_encoder.pt: {e}")
    print("Recreating label encoder manually...")
    # Manually recreate label encoder for 29 signs
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'delete', 'nothing']
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

# Convert to DataFrame
columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
data = pd.DataFrame(x_data.numpy(), columns=columns[:-1])

# Handle y_data as a list of raw labels
if isinstance(y_data, list):
    print(f"y_data is a list with {len(y_data)} elements. Using raw labels...")
    # Map 'del' to 'delete'
    y_data = ['delete' if label == 'del' else label for label in y_data]
    data['label'] = y_data  # Directly use raw labels
else:
    data['label'] = label_encoder.inverse_transform(y_data.numpy())

# Verify all labels are valid
invalid_labels = set(data['label']) - set(labels)
if invalid_labels:
    print(f"Warning: Invalid labels found: {invalid_labels}")
    exit()
else:
    print("All labels are valid.")

# Save
data.to_csv(output_path, index=False)
print(f"Saved {len(data)} rows to {output_path}")