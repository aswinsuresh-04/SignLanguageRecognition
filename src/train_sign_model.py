import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score
import os
import logging
from tqdm import tqdm  # Added for progress bars

logging.basicConfig(filename='training.log', level=logging.INFO)

class SignDataset(Dataset):
    def __init__(self, x_data, y_data, scaler=None):
        self.x_data = x_data
        self.y_data = y_data
        self.scaler = scaler if scaler else StandardScaler()
        self.x_data = self.scaler.fit_transform(self.x_data) if scaler is None else scaler.transform(self.x_data)
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        return torch.tensor(self.x_data[idx], dtype=torch.float32), self.y_data[idx]

class SignClassifier(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def main():
    save_path = r"D:\Study\Project\SignLAnguage\torch_data"
    x_data_path = os.path.join(save_path, 'x_data_balanced.pt')
    y_data_path = os.path.join(save_path, 'y_data_encoded.pt')
    label_encoder_path = os.path.join(save_path, 'label_encoder.pt')

    # Check if files exist
    if not os.path.exists(x_data_path) or not os.path.exists(y_data_path) or not os.path.exists(label_encoder_path):
        print(f"Error: One or more required files missing: {x_data_path}, {y_data_path}, {label_encoder_path}")
        return

    x_data = torch.load(x_data_path, weights_only=False).numpy()
    y_data = torch.load(y_data_path, weights_only=False)
    label_encoder = torch.load(label_encoder_path, weights_only=False)

    from sklearn.model_selection import train_test_split
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler().fit(x_train)
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))

    train_dataset = SignDataset(x_train, y_train, scaler)
    val_dataset = SignDataset(x_val, y_val, scaler)
    test_dataset = SignDataset(x_test, y_test, scaler)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignClassifier(num_classes=29).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_acc = 0
    for epoch in range(30):
        model.train()
        train_loss = 0
        # Add tqdm progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch")
        for data, target in train_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix({"Train Loss": train_loss / (train_bar.n + 1)})
        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        per_class_correct = {i: 0 for i in range(29)}
        per_class_total = {i: 0 for i in range(29)}
        # Add tqdm progress bar for validation
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", unit="batch")
        with torch.no_grad():
            for data, target in val_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                for t, p in zip(target.cpu().numpy(), predicted.cpu().numpy()):
                    per_class_total[t] += 1
                    if t == p:
                        per_class_correct[t] += 1
                val_bar.set_postfix({"Val Loss": val_loss / (val_bar.n + 1)})
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {acc:.2f}%")
        logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {acc:.2f}%")
        for i in range(29):
            class_acc = 100 * per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0
            class_name = label_encoder.inverse_transform([i])[0]
            print(f"  {class_name}: {class_acc:.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
            logging.info(f"  {class_name}: {class_acc:.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
        
        scheduler.step(val_loss)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_path, 'sign_model.pt'))
            print(f"  Saved best model with Val Acc: {acc:.2f}%")
            logging.info(f"  Saved best model with Val Acc: {acc:.2f}%")

    # Test evaluation
    model.eval()
    correct = 0
    total = 0
    per_class_correct = {i: 0 for i in range(29)}
    per_class_total = {i: 0 for i in range(29)}
    # Add tqdm progress bar for testing
    test_bar = tqdm(test_loader, desc="Test", unit="batch")
    with torch.no_grad():
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            for t, p in zip(target.cpu().numpy(), predicted.cpu().numpy()):
                per_class_total[t] += 1
                if t == p:
                    per_class_correct[t] += 1
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    logging.info(f"Test Accuracy: {acc:.2f}%")
    for i in range(29):
        class_acc = 100 * per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0
        class_name = label_encoder.inverse_transform([i])[0]
        print(f"  {class_name}: {class_acc:.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
        logging.info(f"  {class_name}: {class_acc:.2f}% ({per_class_correct[i]}/{per_class_total[i]})")

if __name__ == "__main__":
    main()