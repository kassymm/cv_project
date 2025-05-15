'''
This code trains the models 
'''

import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import load_from_disk
from PIL import Image

# Personal check for GPU
print("CUDA available?", torch.cuda.is_available())
print("Device name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "–")

DATA_PATH = "~/Datasets/cv_project/resized_oxford_pets_224"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

# Load dataset from HuggingFace
dataset = load_from_disk(DATA_PATH)
train_ds = dataset["train"]
test_ds = dataset["test"]

# Define transforms
data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Apply transforms on the fly
def apply_transforms(batch):
    images = [data_transforms(img) for img in batch["image"]]
    batch["image"] = torch.stack(images)
    batch["label"] = torch.tensor(batch["label"])
    return batch

train_ds = train_ds.with_transform(apply_transforms)
test_ds = test_ds.with_transform(apply_transforms)

# DataLoaders
def get_loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=NUM_WORKERS)

train_loader = get_loader(train_ds, shuffle=True)
test_loader  = get_loader(test_ds)

# Determine number of classes
def get_num_classes(ds):
    return len(set(ds["label"]))

num_classes = get_num_classes(train_ds)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models

# MobileNetV2
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(
    mobilenet.classifier[1].in_features, num_classes)
mobilenet = mobilenet.to(device)

# ResNet18 (standard fc)
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = resnet.to(device)

# EfficientNet-B0
effnet = models.efficientnet_b0(pretrained=True)

# replace FC layer with GAP + Linear (classifier)
effnet.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(effnet.classifier[1].in_features, num_classes)
)
effnet = effnet.to(device)

## Training loop

def train_model(model, loader, epochs=NUM_EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

# Train MobileNetV2
if __name__ == "__main__":
    print("Training MobileNetV2...")
    train_model(mobilenet, train_loader)
    
    # print("Training ResNet18...")
    # train_model(resnet, train_loader)
    
    # print("Training EfficientNet-B0...")
    # train_model(effnet, train_loader)
