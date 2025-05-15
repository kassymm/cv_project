'''
This code trains the models 
'''

import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import load_from_disk
from PIL import Image

# # Personal check for GPU
# print("CUDA available?", torch.cuda.is_available())
# print("Device name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "–")

DATA_PATH = "~/Datasets/cv_project/resized_oxford_pets_22"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
LABEL_COLUMN = 'label_cat_dog'


# Load dataset from HuggingFace
dataset = load_from_disk(DATA_PATH)
train_ds = dataset["train"]
test_ds = dataset["test"]

# Build a mapping from each raw label value → integer ID
raw_labels = set(train_ds[LABEL_COLUMN])
label2id   = {lbl: idx for idx, lbl in enumerate(sorted(raw_labels))}
num_classes = len(label2id)
print(f"Found {num_classes} classes: {label2id}")

# Define transforms
data_transforms = transforms.Compose([
    transforms.ToTensor(),   # assumes input is already 224×224 RGB
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# Apply transforms on the fly
def apply_transforms(batch):
    images = [data_transforms(img) for img in batch["image"]]
    batch["image"] = torch.stack(images)
    raw_labels = batch[LABEL_COLUMN]
    # map strings → ints if needed, otherwise assume already numeric
    if label2id is not None:
        labels = [label2id[l] for l in raw_labels]
    else:
        labels = raw_labels
    batch["label"] = torch.tensor(labels, dtype=torch.long)    
    return batch

train_ds = train_ds.map(apply_transforms, batched=True)
test_ds  = test_ds.map(apply_transforms, batched=True)

# DataLoaders
def get_loader(ds, shuffle=False):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,   # ← now 32, not 1
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,         # optional, but common for GPU training
    )

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load models

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
