PARAMS_SOLVER = {'equation': 'Poisson', 'domain': [0., 1.], 'D': 1., 'nx': 101}

PARAMS_METHODS = {'POD': {'method_name': 'POD'},
                  'MLP': {'method_name': 'MLP',
                          'layer_dims': [1, 101],
                          'activations': 'tanh',
                          'device': 'cpu', 'seed': 123},
                  'DEEPONET': {'method_name': 'DEEPONET',
                               'branch': {'layer_dims': [1, 2],
                                          'activations': 'relu', 'device': 'cpu',
                                          'seed': 123},
                               'trunk': {'layer_dims': [1, 2],
                                         'activations': 'tanh' , 'device': 'cpu',
                                         'seed': 123}
                               },
                  'PINN': {'method_name': 'PINN',
                          'layer_dims': [2,  1],
                          'activations': 'tanh',
                          'device': 'cpu', 'seed': 123},
                  'MLPINN': {'method_name': 'MLPINN',
                             'layer_dims': [2, 1],
                             'activations': 'tanh',
                             'device': 'cpu', 'seed': 123},
                  'FNO': {'method_name': 'FNO',
                          'layers_dim': [1, 20, 20, 20, 20],
                          'FourierLayers_modes': 3*[12],
                          'device': 'cpu', 'seed': 123},
                  }

HYPERPARAMS_METHODS = {'POD': {'n_components': 5},
                       'MLP': {'lr': 0.0005289943741212551, 'epochs': 2000, 'optimizer': 'Adam', 'batch_size': 143},
                       'DEEPONET': {'lr': 5.3471990107146325e-05, 'epochs': 50, 'optimizer': 'RMSprop',
                                    'batch_size': 20006, 'device': 'cpu'},
                       'PINN': {'lr': 0.0001858411684138564, 'epochs': 5000, 'optimizer': 'RMSprop', 'batch_size': 28296},
                       'MLPINN': {'lr': 0.0005289943741212551, 'epochs': 10, 'optimizer': 'Adam', 'batch_size': 20000},
                       'FNO': {'lr': 5e-4, 'epochs': 2000, 'optimizer': 'Adam', 'batch_size': 64}
                       }
