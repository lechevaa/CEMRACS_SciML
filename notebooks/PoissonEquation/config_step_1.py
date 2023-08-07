PARAMS_SOLVER = {'equation': 'Poisson', 'domain': [0., 1.], 'D': 1., 'nx': 101}

PARAMS_METHODS = {'POD': {'method_name': 'POD'},
                  'MLP': {'method_name': 'MLP',
                          'layer_dims': [1, 183, 247, 232, 79, 132, 443, 200, 101],
                          'activations': ['relu', 'tanh', 'softplus', 'softplus', 'relu', 'tanh', 'tanh'],
                          'device': 'cpu', 'seed': 123},
                  'DEEPONET': {'method_name': 'DEEPONET',
                               'branch': {'layer_dims': [1,52, 225, 97, 230, 91],
                                          'activations': ['softplus', 'softplus', 'tanh', 'relu'], 'device': 'cpu',
                                          'seed': 123},
                               'trunk': {'layer_dims': [1, 501, 282, 55, 54, 379, 342, 475, 221, 91],
                                         'activations': ['tanh', 'tanh', 'relu', 'tanh', 'tanh', 'tanh', 'relu', 'tanh'] , 'device': 'cpu',
                                         'seed': 123}
                               },
                  'PINN': {'method_name': 'PINN',
                          'layer_dims': [2, 91, 54, 74, 62, 1],
                          'activations': ['tanh', 'softplus', 'softplus', 'relu'],
                          'device': 'cpu', 'seed': 123},
                  'MLPINN': {'method_name': 'MLPINN',
                             'layer_dims': [2, 20, 20, 20, 1],
                             'activations': ['relu', 'tanh', 'softplus'],
                             'device': 'cpu', 'seed': 123},
                  'FNO': {'method_name': 'FNO',
                          'layers_dim': [1, 38, 37, 49, 60, 52],
                          'FourierLayers_modes': [14, 15, 15, 15],
                          'device': 'cpu', 'seed': 123},
                  }

HYPERPARAMS_METHODS = {'POD': {'n_components': 5},
                       'MLP': {'lr': 0.0005289943741212551, 'epochs': 2000, 'optimizer': 'Adam', 'batch_size': 143},
                       'DEEPONET': {'lr': 5.3471990107146325e-05, 'epochs': 50, 'optimizer': 'RMSprop',
                                    'batch_size': 20006, 'device': 'cpu'},
                       'PINN': {'lr': 0.0001858411684138564, 'epochs': 5000, 'optimizer': 'RMSprop', 'batch_size': 28296},
                       'MLPINN': {'lr': 0.0005289943741212551, 'epochs': 10, 'optimizer': 'Adam', 'batch_size': 20000},
                       'FNO': {'lr': 0.0001495740267170668, 'epochs': 2000, 'optimizer': 'RMSprop', 'batch_size': 173}
                       }
