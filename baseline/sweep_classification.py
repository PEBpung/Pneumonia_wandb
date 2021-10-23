import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import model
import sweep_train

from dataload_with_alb  import DiseaseDataset
from torch.utils.data import DataLoader

from pytorch_warmup.warmup_scheduler import GradualWarmupScheduler
import wandb
import config



######### Random seed 고정해주기###########
random_seed = 1234
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
###########################################

def wandb_setting():
    wandb.init(config=config.hyperparameter_defaults)
    w_config = wandb.config

    batch_size= w_config.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ##########################################데이터 로드 하기#################################################
    data_dir = os.path.join(os.getcwd(), "input/RSNA_COVID_512") #train, val 폴더가 들어있는 경로
    classes_name = os.listdir(os.path.join(data_dir, 'train')) #폴더에 들어있는 클래스명
    num_classes =  len(os.listdir(os.path.join(data_dir, 'train'))) #train 폴더 안에 클래스 개수 만큼의 폴더가 있음
    
    datasets = {x: DiseaseDataset(data_dir=os.path.join(data_dir, x), img_size=224, bit=8, data_type='img', mode= x) for x in ['train', 'val']}
    sub_dataset = {x: torch.utils.data.Subset(datasets[x], indices=range(0, len(datasets[x]), 5)) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(sub_dataset[x], batch_size=batch_size, shuffle=w_config.shuffle, num_workers=5) for x in ['train', 'val']}
    dataset_sizes = {x: len(sub_dataset[x]) for x in ['train', 'val']}
    num_iteration = {x: np.ceil(dataset_sizes[x] / batch_size) for x in ['train', 'val']}
    #############################################################################################################################

    num_classes =  len(os.listdir(os.path.join(data_dir, 'train'))) #train 폴더 안에 클래스 개수 만큼의 폴더가 있음
    
    if w_config.model == 'resnet':
        net = model.PneumoniaNet(img_channel=1, num_classes=num_classes) # pretrained 모델 사용
    elif w_config.model == 'custom':
        net = model.ResNet50(img_channel=1, num_classes=num_classes) #gray scale = 1, color scale =3
    elif w_config.model == 'effnet':
        net = model.Efficient(img_channel=1, num_classes=num_classes)


    net = net.to(device) #딥러닝 모S델 GPU 업로드

    criterion = nn.CrossEntropyLoss() #loss 형태 정해주기

    if w_config.optimizer == 'sgd':
        optimizer_ft = optim.SGD(net.parameters(), lr=w_config.learning_rate, momentum= w_config.momentum)# optimizer 종류 정해주기
    elif w_config.optimizer == 'adam':
        optimizer_ft = optim.Adam(net.parameters(), lr=0.0001)

    #Learning rate scheduler: Warm-up with ReduceLROnPlateau
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft, mode='min', factor=0.5, patience=10)
    scheduler_warmup = GradualWarmupScheduler(optimizer_ft, multiplier=1, total_epoch=5, after_scheduler=scheduler_lr)

    wandb.watch(net, log='all') # wandb에 남길 log 기록하기, all은 파라미터와 gradient 확인
    sweep_train.train_model(dataloaders, dataset_sizes, num_iteration, net, criterion, optimizer_ft, scheduler_warmup ,device, classes_name, wandb, num_epoch=20)

    #model_ft = sweep_train.train_model(dataloaders, dataset_sizes, num_iteration, net, criterion, optimizer_ft, scheduler_warmup,  device, wandb, num_epoch=30)

sweep_id = wandb.sweep(config.sweep_config, project="covid_v1", entity="pebpung")
wandb.agent(sweep_id, wandb_setting, count=3)





