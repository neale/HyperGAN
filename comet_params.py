def get_hyper_params():
    hyperparameters = {
            'network type': 'conv',
            'param size': (128, 3, 3),
            'batch_size': 64, 
            'grad penalty': 10, 
            'epochs': 200000,
            'optim': 'adam',
            'dataset': 'cifar10'
    }
    return hyperparameters


