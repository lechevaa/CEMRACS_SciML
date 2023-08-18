import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PARAMS_SOLVER = {'equation': 'Poisson', 'domain': [0., 1.], 'D': None, 'nx': 101}

PARAMS_METHODS = {'POD': {'method_name': 'POD'},
                  'MLP': {'method_name': 'MLP',
                          'layer_dims': [3, 21, 105, 119, 21, 69, 55, 1],
                          'activations': ['relu', 'relu', 'relu', 'tanh', 'tanh', 'relu'],
                          'device': device, 'seed': 123},
                  'DEEPONET': {'method_name': 'DEEPONET',
                               'branch': {'layer_dims': [202, 525, 265, 400, 215],
                                          'activations': ['relu', 'relu', 'tanh'], 'device': device,
                                          'seed': 123},
                               'trunk': {'layer_dims': [1, 868, 434, 552, 929, 756, 215] ,
                                         'activations': ['tanh', 'relu', 'relu', 'tanh', 'tanh'], 'device': device,
                                         'seed': 123}
                               },
                  'PINN': {'method_name': 'PINN',
                          'layer_dims': [2, 21, 30, 1] ,
                          'activations': 'tanh',
                          'device': device, 'seed': 123},
                  'MLPINN': {'method_name': 'MLPINN',
                             'layer_dims': [2, 17, 15, 26, 31, 1],
                             'activations': 'tanh',
                             'device': device, 'seed': 123},
                  'FNO': {'method_name': 'FNO',
                          'layers_dim': [2, 55, 32, 50, 37, 42],
                          'FourierLayers_modes': [14, 12, 15, 15],
                          'device': device, 'seed': 123},
                  }


n_epochs = 10000
HYPERPARAMS_METHODS = {'POD': {'n_components': 10},
                       'MLP': {'lr': 3e-4, 'epochs': n_epochs, 'optimizer': 'Adam'},
                       'DEEPONET': {'lr': 1e-03, 'epochs': n_epochs, 'optimizer': 'Adam',
                                    'device': device},
                       'PINN': {'lr': 4e-3, 'epochs': n_epochs, 'optimizer': 'Adam'},
                       'MLPINN': {'lr': 1e-4, 'epochs': n_epochs, 'optimizer': 'Adam'},
                       'FNO': {'lr': 4e-4, 'epochs': n_epochs, 'optimizer': 'Adam', 'batch_size': 476}
                       }
