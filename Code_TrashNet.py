import os
# Function to count the number of images in a folder
def count_images_in_folder(folder_path):
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                image_count += 1
    return image_count

# Function to get the image count for each subfolder within the parent folder
def get_image_counts_in_folders(parent_folder_path):
    for folder_name in os.listdir(parent_folder_path):
        folder_path = os.path.join(parent_folder_path, folder_name)
        if os.path.isdir(folder_path):
            image_count = count_images_in_folder(folder_path)
            print(f"Folder '{folder_name}' contains {image_count} images.")

# Replace 'PARENT_FOLDER_PATH' with the path of your parent folder
parent_folder_path = 'E:/5-1/2022-Research/TrashNet/TrashData'

# Call the function to get image counts in each subfolder
get_image_counts_in_folders(parent_folder_path)



####################################################################3



import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Set random seeds for reproducibility
seed = 25
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


# Paths

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

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
picture_saving_path = f'E:/5-1/2022-Research/TrashNet/Figure'
model_directory = f'E:/5-1/2022-Research/TrashNet/Model'
result_path = f'E:/5-1/2022-Research/TrashNet/result'







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
    image_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
    label = class_to_label[class_name]
    all_image_paths.extend(image_paths)
    all_labels.extend([label] * len(image_paths))

dataset = CustomDataset(all_image_paths, all_labels, transform=transform)

# Split dataset into training and validation sets (80:20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size =64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model training and evaluation
num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)



# Define the attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
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

# Load pretrained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)

# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# Modify classifier for 6 classes
num_features = model.classifier[1].in_features

# Add attention mechanism before the classifier
class MobileNetV2WithAttention(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2WithAttention, self).__init__()
        self.features = model.features
        self.attention = SelfAttention(in_dim=1280)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

model = MobileNetV2WithAttention(num_classes=num_classes).to(device)

model = model.to(device)
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
num_epochs = 30   # Number of epochs
max_val_acc=0
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
    if val_accuracy>max_val_acc :
      # Save the model
      model_path = os.path.join(model_directory, 'mobilenet_v2_trashnet.pth')
      torch.save(model.state_dict(), model_path)
      print(f"Model has been saved to {model_path}")
      max_val_acc=val_accuracy



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
    print(f"Accuracy and loss curves have been saved to {accuracy_loss_plot_path}")

plt.close()

# Compute and plot ROC AUC curves for each class
# Compute and plot ROC AUC curves for each class
all_labels_binarized = label_binarize(all_labels, classes=list(range(num_classes)))

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(all_labels_binarized[:, i], np.array(all_probs)[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")

# Save ROC AUC curves in different formats
for ext in ['png', 'eps']:
    roc_auc_plot_path = os.path.join(picture_saving_path, f'roc_auc_curve.{ext}')
    plt.savefig(roc_auc_plot_path, format=ext)
    print(f"ROC AUC curves have been saved to {roc_auc_plot_path}")
plt.close()
print("All plots have been saved in specified formats.")