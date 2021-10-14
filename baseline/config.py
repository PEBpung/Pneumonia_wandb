import math

hyperparameter_defaults  = {
        'epochs': 2,
        'batch_size': 128,
        'fc_layer_size': 128,
        'weight_decay': 0.0005,
        'learning_rate': 1e-3,
        'activation': 'relu',
        'optimizer': 'adam',
        'seed': 42
    }

sweep_config = {
    'method': 'random',
    'metric' : {
        'name': 'loss',
        'goal': 'minimize'   
        },
    'parameters' : {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'dropout': {
            'values': [0.3, 0.4, 0.5]
            },
        'epochs': {
            'values': [3]
            },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            },
        'batch_size': {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(16),
            'max': math.log(32),
            },
        'data_augmentation1': {
            'values': ['brightness', 'no_aug']
        },
        'data_augmentation2': {
            'values': ['contrast', 'no_aug']
        },
    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }