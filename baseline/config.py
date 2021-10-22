import math

hyperparameter_defaults  = {
        'epochs': 2,
        'batch_size': 8,
        'fc_layer_size': 128,
        'weight_decay': 0.0005,
        'learning_rate': 1e-3,
        'activation': 'relu',
        'optimizer': 'adam',
        'seed': 42
    }

sweep_config = {
    'name': 'bayes_custom_model',
    'method': 'bayes',
    'project': "pebpung_v1", 
    'entity': 'pebpung',
    'metric' : {
        'name': 'val_loss',
        'goal': 'minimize'   
        },
    'parameters' : {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'shuffle': {
            'values': [True, False]
            },
        'momentum': {
            'values' : [0.1, 0.3, 0.5, 0.9, 0.99]
        },
        ##여러 줄 주석 추가: ctrl+K+C 동시에 누르기
        ##여러줄 주석 해제: ctrl+K+U 동시에 누르기
        # 'dropout': {
        #     'values': [0.3, 0.4, 0.5]
        #     },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.0001
            },
        # 'batch_size': {
        #     'distribution': 'q_log_uniform',
        #     'q': 1,
        #     'min': math.log(16),
        #     'max': math.log(32),
        #     },
        # 'data_augmentation1': {
        #     'values': ['brightness', 'no_aug']
        # },
        # 'data_augmentation2': {
        #     'values': ['contrast', 'no_aug']
        # },
    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }