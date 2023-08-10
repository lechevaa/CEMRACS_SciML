import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PARAMS_SOLVER = {'equation': 'Poisson', 'domain': [0., 1.], 'D': 1., 'nx': 101}

PARAMS_METHODS = {'POD': {'method_name': 'POD'},
                  'MLP': {'method_name': 'MLP',
                          'layer_dims': [2, 30, 23, 20, 19, 32, 1] ,
                          'activations': 'tanh',
                          'device': device, 'seed': 123},
                  'DEEPONET': {'method_name': 'DEEPONET',
                               'branch': {'layer_dims': [101, 22, 9, 21, 32, 19],
                                          'activations': 'tanh', 'device': 'cpu',
                                          'seed': 123},
                               'trunk': {'layer_dims': [1, 13, 19],
                                         'activations': 'tanh', 'device': device,
                                         'seed': 123}
                               },
                  'PINN': {'method_name': 'PINN',
                          'layer_dims': [2, 17, 15, 26, 31, 1],
                          'activations': 'tanh',
                          'device': device, 'seed': 123},
                  'MLPINN': {'method_name': 'MLPINN',
                             'layer_dims': [2, 17, 15, 26, 31, 1],
                             'activations': 'tanh',
                             'device': device, 'seed': 123},
                  'FNO': {'method_name': 'FNO',
                          'layers_dim': [1, 33, 45, 38, 49],
                          'FourierLayers_modes': [15, 12, 14],
                          'device': device, 'seed': 123},
                  }

HYPERPARAMS_METHODS = {'POD': {'n_components': 10},
                       'MLP': {'lr': 2e-4, 'epochs': 10, 'optimizer': 'Adam'},
                       'DEEPONET': {'lr': 4e-04, 'epochs': 10, 'optimizer': 'Adam',
                                    'device': 'cpu'},
                       'PINN': {'lr': 2e-4, 'epochs': 10, 'optimizer': 'Adam'},
                       'MLPINN': {'lr': 2e-4, 'epochs': 10, 'optimizer': 'Adam'},
                       'FNO': {'lr': 8e-4, 'epochs': 10, 'optimizer': 'Adam', 'batch_size': 175}
                       }
