import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='emotion_training.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Custom Dataset for FER-2013
class FER2013Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for cls in classes:
            cls_path = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_path):
                self.images.append(os.path.join(cls_path, img_name))
                self.labels.append(self.class_to_idx[cls])
        
        # Log class distribution
        for cls, idx in self.class_to_idx.items():
            count = self.labels.count(idx)
            logging.info(f"Class: {cls}, Images: {count}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

# CNN Model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Data Augmentation with emphasis on 'sad'
transform_train = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Check device and print
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

if __name__ == '__main__':
    # Paths
    save_path = r"D:\Study\Project\SignLanguageRecognition\torch_data"
    model_path = os.path.join(save_path, 'emotion_model.pt')
    train_dir = os.path.join(r"D:\Study\Project\SignLanguageRecognition\data\fer2013", 'train')
    test_dir = os.path.join(r"D:\Study\Project\SignLanguageRecognition\data\fer2013", 'test')

    # Load datasets
    num_workers = 0
    train_dataset = FER2013Dataset(train_dir, transform=transform_train)
    test_dataset = FER2013Dataset(test_dir, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Compute class weights with emphasis on 'sad' and 'disgust'
    labels = train_dataset.labels
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights[5] *= 1.5  # Boost 'sad' weight
    class_weights[1] *= 1.5  # Boost 'disgust' weight
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Initialize model, loss, and optimizer
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # Training loop with 100 epochs and early stopping
    num_epochs = 100
    best_val_acc = 0.0
    patience = 10  # Stop if no improvement for 10 epochs
    trigger_times = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase with progress bar
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            train_progress.set_postfix({
                'loss': f"{running_loss / (train_progress.n + 1):.4f}",
                'acc': f"{train_acc:.2f}%"
            })
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation phase with progress bar
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_progress = tqdm(test_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(test_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

        logging.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, "
                     f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with val accuracy: {best_val_acc:.2f}%")
            logging.info(f"New best model saved to {model_path} with val accuracy: {best_val_acc:.2f}%")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1} with best val accuracy: {best_val_acc:.2f}%")
                logging.info(f"Early stopping triggered at epoch {epoch+1} with best val accuracy: {best_val_acc:.2f}%")
                break

    print(f"\nTraining complete. Best model saved as {model_path} with val accuracy: {best_val_acc:.2f}%")