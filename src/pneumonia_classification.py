from __future__ import print_function, division

from model import PneumoniaNet
from dataset import XrayDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import time
import glob 
import copy
import random

plt.ion()   # interactive mode

random_seed = 1234
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else  'cpu'

train_normal = glob.glob("input/chest_xray/train/NORMAL/*")
train_pneumonia = glob.glob("input/chest_xray/train/PNEUMONIA/*")

valid_normal = glob.glob("input/chest_xray/val/NORMAL/*")
valid_pneumonia = glob.glob("input/chest_xray/val/PNEUMONIA/*")

test_normal = glob.glob("input/chest_xray/test/NORMAL/*")
test_pneumonia = glob.glob("input/chest_xray/test/PNEUMONIA/*")

train_paths = train_normal + train_pneumonia
valid_paths = valid_normal + valid_pneumonia
test_paths = test_normal + test_pneumonia

train_labels = [0] * len(train_normal) + [1] * len(train_pneumonia)
valid_labels = [0] * len(valid_normal) + [1] * len(valid_pneumonia)
test_labels = [0] * len(test_normal) + [1] * len(test_pneumonia)

image_size = (224, 224)

train_transform = A.Compose( [
    A.Resize(*image_size),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RandomCrop(height=128, width=128),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()]
)

val_transform = A.Compose([
    A.Resize(*image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()]
)

train_dataset = XrayDataset(train_paths, train_labels, train_transform)
valid_dataset = XrayDataset(valid_paths, valid_labels, val_transform)

lr = 3e-3

num_epochs = 5
train_batch_size = 16
valid_batch_size = 16


train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=5, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, num_workers=5, shuffle=False)


dataloaders = {
    "train": train_dataloader,
    "valid": valid_dataloader
}

logging_steps = {
    "train": len(dataloaders["train"]) // 10,
    "valid": len(dataloaders["valid"]) # // 10
}

dataset_sizes = {
    "train": len(train_dataset),
    "valid": len(valid_dataset)
}

batch_sizes = {
    "train": train_batch_size,
    "valid": valid_batch_size
}

pretrained = True


def train_model(model, criterion, optimizer, num_epochs, device="cuda"):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    
    for epoch in tqdm(range(num_epochs), leave=False):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    
                    preds = outputs.sigmoid() > 0.5
                    loss = criterion(outputs, labels.float())
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if (i % logging_steps[phase] == 0) & (i > 0):
                    avg_loss = running_loss / ((i+1) * batch_sizes[phase])
                    avg_acc = running_corrects / ((i+1) * batch_sizes[phase])   
                    
                    print(f"EPOCH {epoch+1} / {num_epochs} | loss : {avg_loss:.3f} | acc : {avg_acc:.3f}")
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            
            print(f"[{phase}]:::::::: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            
            
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
    time_elapsed = time.time() - since
    
    print(f"training took {time_elapsed} seconds")
    
    model.load_state_dict(best_model_wts)
    
    return model


model = PneumoniaNet()
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
model = train_model(model, criterion, optimizer, num_epochs, device)
