import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

import numpy as np
from PIL import Image
import os

import wandb
import config

random_seed = 3
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
device = 'cuda' if torch.cuda.is_available() else  'cpu'

class chest_xray_data(object):
    def __init__(self, root, split, transforms=None):
        self.root = root
        self.transforms = transforms
        self.data_dir = os.path.join(root, split)
        self.imgs = [(i, 'NORMAL') for i in list(sorted(os.listdir(os.path.join(self.data_dir, 'NORMAL'))))]
        self.imgs += [(i, 'PNEUMONIA') for i  in list(sorted(os.listdir(os.path.join(self.data_dir, 'PNEUMONIA'))))]
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.imgs[idx][1], self.imgs[idx][0])
        img = Image.open(img_path).convert('RGB').resize((448, 448))
        img = np.moveaxis(np.array(img)/255.0, -1, 0)
        label = 0 if self.imgs[idx][1] == 'NORMAL' else 1
        if self.transforms is not None:
            img, label = self.transforms(img, label)
        return img, label

def train_epoch(dataloader, model, loss_fn, optimizer, device, wandb, num_epoch):
    for _ in range(num_epoch):
        size = len(dataloader)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
        
            pred = model(X.float())
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % int(size/10) == 0:
                loss, current = loss.item(), batch
                ######## wandb log ########
                wandb.log({"loss": loss})
                print(f"loss: {loss:>7f}  [{current:>2d}/{size:>2d}]")


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 2)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    wandb.init(config=config.hyperparameter_defaults)

    ######## wandb initialize ########
    w_config = wandb.config
    root = "input/chest-xray-pneumonia"

    train_data = chest_xray_data(root, 'train')
    #test_data = chest_xray_data(root, 'test')

    train_dataloader = DataLoader(train_data, w_config.batch_size, True)
    #test_dataloader = DataLoader(test_data, 128, True)
    
    fully_connected = FullyConnected().to(device)
    model_conv = models.resnet18(pretrained=True)
    model_conv = model_conv.to(device)
    for param in model_conv.parameters():
        param.requires_grad = False
    model = nn.Sequential(model_conv, fully_connected)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=w_config.learning_rate)

    ######## wandb watch ########
    wandb.watch(model, log='all')
    
    train_epoch(train_dataloader, model, criterion, optimizer, device, wandb, num_epoch=20)

######## wandb sweep & agent ########
sweep_id = wandb.sweep(config.sweep_config, project='pneumonia_sweep', entity='pebpung')
wandb.agent(sweep_id, train, count=20)