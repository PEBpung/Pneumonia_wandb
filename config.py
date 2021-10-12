import math
import torch
from torchvision import transforms

hyperparameter_defaults  = {
        'batch_size': 128,
        'learning_rate': 1e-2,
        'seed': 42
    }

sweep_config = {
    'method': 'random',
    'project': "sweeps-test3", 
    'entity': 'pebpung',
    'metric' : {
        'name': 'loss',
        'goal': 'minimize'   
        },
    'parameters' : {
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.001
            },
        'batch_size': {
            'values': [128, 256]
            }
        },
    }

train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])