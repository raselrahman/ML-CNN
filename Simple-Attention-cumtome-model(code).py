import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import pandas as pd

# Set random seeds for reproducibility
seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label

# Paths
data_dir = 'E:/5-1/2022-Research/TrashNet/TrashData'
picture_saving_path = 'E:/5-1/2022-Research/TrashNet/Figure'
model_directory = 'E:/5-1/2022-Research/TrashNet/Model'
result_path = 'E:/5-1/2022-Research/TrashNet/result'

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_names = os.listdir(data_dir)
class_to_label = {class_name: label for label, class_name in enumerate(class_names)}

all_image_paths = []
all_labels = []

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    image_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label = class_to_label[class_name]
    all_image_paths.extend(image_paths)
    all_labels.extend([label] * len(image_paths))

dataset = CustomDataset(all_image_paths, all_labels, transform=transform)

# Split dataset into training and validation sets (80:20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model definition with self-attention
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class AttentionModel(nn.Module):
    def __init__(self, num_classes):
        super(AttentionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SelfAttention(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_names)
model = AttentionModel(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Arrays to store metrics
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
train_precision = []
val_precision = []
train_recall = []
val_recall = []
train_kappa = []
val_kappa = []

# Training loop
num_epochs = 30
max_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    if val_accuracy > max_val_acc:
        model_path = os.path.join(model_directory, 'self_attention_trashnet.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model has been saved to {model_path}")
        max_val_acc = val_accuracy

    # Calculate precision, recall, and kappa scores
    train_precision.append(precision_score(all_labels, all_preds, average='macro'))
    val_precision.append(precision_score(all_labels, all_preds, average='macro'))
    train_recall.append(recall_score(all_labels, all_preds, average='macro'))
    val_recall.append(recall_score(all_labels, all_preds, average='macro'))
    train_kappa.append(cohen_kappa_score(all_labels, all_preds))
    val_kappa.append(cohen_kappa_score(all_labels, all_preds))

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save metrics to a DataFrame
metrics_df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Train Accuracy': train_accuracies,
    'Val Accuracy': val_accuracies,
    'Train Loss': train_losses,
    'Val Loss': val_losses,
    'Train Precision': train_precision,
    'Val Precision': val_precision,
    'Train Recall': train_recall,
    'Val Recall': val_recall,
    'Train Kappa': train_kappa,
    'Val Kappa': val_kappa
})

# Save metrics to a CSV file
metrics_csv_path = os.path.join(result_path, 'metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)

print("Training complete. Metrics have been saved.")

# Plot accuracy and loss curves
plt.figure(figsize=(14, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(metrics_df['Epoch'], metrics_df['Train Accuracy'], label='Train Accuracy')
plt.plot(metrics_df['Epoch'], metrics_df['Val Accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validation Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(metrics_df['Epoch'], metrics_df['Train Loss'], label='Train Loss')
plt.plot(metrics_df['Epoch'], metrics_df['Val Loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')

# Save accuracy and loss curves in different formats
for ext in ['png', 'eps']:
    accuracy_loss_plot_path = os.path.join(picture_saving_path, f'accuracy_loss_curve.{ext}')
    plt.savefig(accuracy_loss_plot_path, format=ext)
    print(f"Accuracy and loss curves saved as {ext}.")
