PARAMS_SOLVER = {'equation': 'Poisson', 'domain': [0., 1.], 'D': 1., 'nx': 101}

PARAMS_METHODS = {'POD': {'method_name': 'POD'},
                  'MLP': {'method_name': 'MLP',
                          'layer_dims': [1, 183, 247, 232, 79, 132, 443, 200, 101],
                          'activations': ['relu', 'tanh', 'softplus', 'softplus', 'relu', 'tanh', 'tanh'],
                          'device': 'cpu', 'seed': 123}, 
                 'DEEPONET': {'method_name': 'DEEPONET', 
                              'branch': {'layer_dims': [1, 20, 10], 'activations': 'tanh', 'device': 'cpu', 'seed': 123}, 
                              'trunk':{'layer_dims': [1, 20, 10], 'activations': 'tanh', 'device': 'cpu', 'seed': 123}
                             },
                  'PINN': {'method_name': 'PINN',
                          'layer_dims': [2, 183, 247, 232, 79, 132, 443, 200, 1],
                          'activations': ['relu', 'tanh', 'softplus', 'softplus', 'relu', 'tanh', 'tanh'],
                          'device': 'cpu', 'seed': 123},
                  'MLPINN': {'method_name': 'MLPINN',
                          'layer_dims': [2, 183, 247, 232, 79, 132, 443, 200, 1],
                          'activations': ['relu', 'tanh', 'softplus', 'softplus', 'relu', 'tanh', 'tanh'],
                          'device': 'cpu', 'seed': 123}, 
                 }

HYPERPARAMS_METHODS = {'POD': {'n_components': 5},
                       'MLP': {'lr': 0.0005289943741212551, 'epochs': 10, 'optimizer': 'Adam', 'batch_size': 143},
                       'DEEPONET': {'lr': 0.0005984992954500306, 'epochs': 10, 'optimizer': 'Adam', 'batch_size': 256, 'device': 'cpu'},
                       'PINN': {'lr': 0.0005289943741212551, 'epochs': 10, 'optimizer': 'Adam', 'batch_size': 143},
                       'MLPINN': {'lr': 0.0005289943741212551, 'epochs': 10, 'optimizer': 'Adam', 'batch_size': 143}
                      }
