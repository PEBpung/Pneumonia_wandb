import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

from torchvision import models

import time
import os
import copy
import model

import random

from dataload_with_alb import DiseaseDataset
from torch.utils.data import DataLoader

from pytorch_warmup.warmup_scheduler import GradualWarmupScheduler


#########Random seed 고정해주기###########
random_seed = 3407
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
###########################################


##########################################데이터 로드 하기#################################################
data_dir = os.path.join(os.getcwd(), "input/RSNA_COVID_512")
print(data_dir)
batch_size= 8
"""
dataset_train = DiseaseDataset(data_dir=os.path.join(data_dir, 'train'), img_size=224, bit=8, data_type='img', mode='train')
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)

dataset_val = DiseaseDataset(data_dir=os.path.join(data_dir, 'val'), img_size=224, bit=8, data_type='img', mode='val')
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

dataloaders = {
    "train": train_loader,
    "val": val_loader
}

#↓↓Data 총 개수 (train 폴더 데이터 개수 or val 폴더 데이터 개수 ← batch 계산 용)
dataset_sizes = {
    "train": len(dataset_train),
    "val": len(dataset_val)
}

#↓↓Total number of iterations per an epoch
num_iteration = {
    "train": np.ceil(dataset_sizes["train"] / batch_size),
    "val": np.ceil(dataset_sizes["val"] / batch_size)
}

"""

datasets = {x: DiseaseDataset(data_dir=os.path.join(data_dir, x), img_size=512, bit=8, data_type='img', mode= x ) for x in ['train', 'val']}
dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=False, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
num_iteration = {x: np.ceil(dataset_sizes[x] / batch_size) for x in ['train', 'val']}

#############################################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(net, criterion, optim, scheduler, num_epoch):
    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 100

    for epoch in range(num_epoch):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            loss_arr = []
            running_corrects = 0
            running_loss = 0

            #train dataset 로드하기
            for iteration_th, (inputs, labels) in enumerate(dataloaders[phase]): #iteration_th: 몇 번재 iteration 인지 알려 줌 "ex) batch_th=0 ← 첫 번째 batch 시작"
                
                ###########################GPU에 데이터 업로드##########################
                inputs = inputs.to(device) #image 데이터 GPU에 업로드
                labels = labels.to(device) #label 데이터 GPU에 업로드 // {labels: 0 → Normal, labels: 1 → Pneumonia} <== alb_data_load_classification.py 참고
                ########################################################################

                # backward pass ← zero the parameter gradients
                optim.zero_grad()

                
                with torch.set_grad_enabled(phase == "train"): # track history if only in train
                    outputs = net(inputs) #output 결과값은 softmax 입력 직전의 logit 값들
                    _, preds = torch.max(outputs, 1) #pred: 0 → Normal <== labels 참고
                    #preds2 = outputs.sigmoid() > 0.5
                    loss = criterion(outputs, labels) #criterion에 output이 들어가면 softmax 이 후의 확률 값으로 변하고, 변환된 확률 값과 label을 비교하여 loss 계산

                    loss_arr += [loss.item()] #Iteration 당 Loss 계산

                    if phase == "train":
                        loss.backward() #계산된 loss에 의해 backward (gradient) 계산
                        optim.step() #계산된 gradient를 참고하여 backpropagation으로 update
                        
                        print("TRAIN: EPOCH %04d / %04d | ITERATION %04d / %04d | LOSS %.4f" %
                        (epoch + 1, num_epoch, iteration_th, num_iteration['train'], np.mean(loss_arr)))

                    elif phase == 'val':
                        print()
                        print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                                (epoch + 1, num_epoch, num_iteration['val'], num_iteration['val'], np.mean(loss_arr))) 

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step_ReduceLROnPlateau(np.mean(loss_arr)) #learning rate scheduler 실행

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('Epoch {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())

        print()
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net

####train 폴더 안에 클래스 개수 만큼의 폴더가 있음######
num_classes =  len(os.listdir(os.path.join(data_dir, 'train'))) 

net = model.PneumoniaNet(img_channel=1, num_classes=num_classes)
# net = model.ResNet50(img_channel=1, num_classes=num_classes)

#딥러닝 모델 GPU 업로드
net = net.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized


# optimizer_ft = optim.Adam(net.parameters(), lr=0.0001)

optimizer_ft = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

# Decay
scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft,
                                         mode='min',
                                         factor=0.5,
                                         patience=10,)

scheduler_warmup = GradualWarmupScheduler(optimizer_ft, multiplier=1, total_epoch=5, after_scheduler=scheduler_lr)

model_ft = train_model(net, criterion, optimizer_ft, scheduler_warmup, num_epoch=3)
