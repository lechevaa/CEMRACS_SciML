PARAMS_SOLVER = {'equation': 'Poisson', 'domain': [0., 1.], 'D': 1., 'nx': 101}

PARAMS_METHODS = {'POD': {'method_name': 'POD'},
                  'MLP': {'method_name': 'MLP',
                          'layer_dims': [1, 391, 141, 247, 114, 349, 101],
                          'activations': ['relu', 'tanh', 'softplus', 'relu', 'relu'],
                          'device': 'cpu', 'seed': 123}, 
                 'DEEPONET': {'method_name': 'DEEPONET', 
                              'branch': {'layer_dims': [1, 20, 20], 'activations': 'tanh', 'device': 'cpu', 'seed': 123}, 
                              'trunk':{'layer_dims': [1, 20, 20], 'activations': 'tanh', 'device': 'cpu', 'seed': 123}
                             }
                 }

HYPERPARAMS_METHODS = {'POD': {'n_components': 5},
                       'MLP': {'lr': 0.0005984992954500306, 'epochs': 100, 'optimizer': 'Adam', 'batch_size': 256},
                       'DEEPONET': {'lr': 0.0005984992954500306, 'epochs': 100, 'optimizer': 'Adam', 'batch_size': 256, 'device': 'cpu'} }
