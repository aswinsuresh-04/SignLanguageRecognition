import torch
from sklearn.preprocessing import LabelEncoder
import os

def encode_labels():
    # Paths
    save_path = r"D:\Study\Project\SignLAnguage\torch_data"
    y_data_path = os.path.join(save_path, 'y_data.pt')
    y_data_encoded_path = os.path.join(save_path, 'y_data_encoded.pt')
    label_encoder_path = os.path.join(save_path, 'label_encoder.pt')

    # Load string labels
    y_data = torch.load(y_data_path, weights_only=False)  # List of strings, e.g., ["A", "A", "B", ...]

    # Encode labels
    le = LabelEncoder()
    y_data_encoded = le.fit_transform(y_data)  # Convert to integers, e.g., [0, 0, 1, ...]

    # Save encoded labels and label encoder
    torch.save(torch.tensor(y_data_encoded, dtype=torch.long), y_data_encoded_path)
    torch.save(le, label_encoder_path)

    print(f"✅ Encoded labels saved to {y_data_encoded_path}")
    print(f"✅ Label encoder saved to {label_encoder_path}")
    print(f"Classes: {le.classes_}")

if __name__ == "__main__":
    encode_labels()