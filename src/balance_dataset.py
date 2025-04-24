import torch
import os
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def main():
    save_path = r"D:\Study\Project\SignLAnguage\torch_data"
    x_data_path = os.path.join(save_path, 'x_data.pt')
    y_data_path = os.path.join(save_path, 'y_data.pt')

    # Check if input files exist
    if not os.path.exists(x_data_path) or not os.path.exists(y_data_path):
        print(f"Error: {x_data_path} or {y_data_path} not found.")
        return

    # Load data
    x_data = torch.load(x_data_path, weights_only=False)
    y_data = torch.load(y_data_path, weights_only=False)
    print(f"Loaded {len(x_data)} samples. Label counts: {Counter(y_data)}")

    # Balance dataset
    label_counts = Counter(y_data)
    max_samples = 3000  # Cap samples per class
    balanced_x, balanced_y = [], []
    for label in label_counts:
        indices = [i for i, y in enumerate(y_data) if y == label]
        sample_count = len(indices)
        if sample_count > max_samples:
            indices = random.sample(indices, max_samples)
        elif sample_count < max_samples:
            print(f"Warning: Class {label} has only {sample_count} samples, less than {max_samples}.")
            if label == "nothing" and sample_count < 100:
                print(f"Critical: 'nothing' has too few samples ({sample_count}). Collect more data.")
        for i in indices:
            balanced_x.append(x_data[i])
            balanced_y.append(y_data[i])
    
    # Convert to tensors
    balanced_x = torch.stack(balanced_x)
    print(f"Balanced dataset: {len(balanced_x)} samples. Label counts: {Counter(balanced_y)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_data_encoded = label_encoder.fit_transform(balanced_y)
    y_data_encoded = torch.tensor(y_data_encoded, dtype=torch.long)

    # Save balanced data and label encoder
    torch.save(balanced_x, os.path.join(save_path, 'x_data_balanced.pt'))
    torch.save(y_data_encoded, os.path.join(save_path, 'y_data_encoded.pt'))
    torch.save(label_encoder, os.path.join(save_path, 'label_encoder.pt'))
    print(f"Saved x_data_balanced.pt, y_data_encoded.pt, and label_encoder.pt to {save_path}")

if __name__ == "__main__":
    main()