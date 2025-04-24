import torch
import os
from collections import Counter

def inspect_labels():
    # Paths
    save_path = r"D:\Study\Project\SignLAnguage\torch_data"
    y_data_path = os.path.join(save_path, 'y_data.pt')
    y_data_encoded_path = os.path.join(save_path, 'y_data_encoded.pt')

    # Load and inspect y_data (string labels)
    y_data = torch.load(y_data_path, weights_only=False)
    print("String Labels (y_data.pt):")
    print(f"  Total samples: {len(y_data)}")
    print(f"  Unique labels: {set(y_data)}")
    print(f"  Label counts: {Counter(y_data)}")

    # Load and inspect y_data_encoded (integer labels)
    y_data_encoded = torch.load(y_data_encoded_path, weights_only=True)
    print("\nEncoded Labels (y_data_encoded.pt):")
    print(f"  Total samples: {len(y_data_encoded)}")
    print(f"  Unique labels: {set(y_data_encoded.tolist())}")
    print(f"  Label range: min={y_data_encoded.min().item()}, max={y_data_encoded.max().item()}")
    print(f"  Label counts: {Counter(y_data_encoded.tolist())}")

if __name__ == "__main__":
    inspect_labels()