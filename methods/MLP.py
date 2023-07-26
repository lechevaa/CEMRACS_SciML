import torch
from typing import Dict


class MLP(torch.nn.Module):
    def __init__(self, params: Dict):
        self._params = params
        self._solver_params = params['solver']
        self._method_params = params['method']

        super().__init__()

        if len(params['method']['layer_dims']) < 3:
            raise Exception('len(layers_dim) < 3')
        else:
            self._layers_dim = params['method']['layer_dims']

        if isinstance(params['method']['activations'], list) and len(params['method']['activations']) == len(params['method']['layer_dims']) - 2:
            self._activations = params['method']['activations']
        elif isinstance(params['method']['activations'], str):
            self._activations = (len(params['method']['layer_dims']) - 2) * [params['method']['activations']]
        else:
            raise Exception(f'Number of activations function not consistent with architecture: '
                            f'got {len(params["method"]["activations"])}, expected {len(params["method"]["layer_dims"])-2} or {1}')

        self._layers = self.create_net()
        if 'seed' in params.keys():
            self.init_net(params['method']['seed'])
        else:
            self.init_net(123)

        if 'device' in params.keys():
            if params['method']['device'] == 'cpu':
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        print(self)
        print('device :', self.device)

    def create_net(self):
        layers = torch.nn.ModuleList()
        dim = self._layers_dim[0]
        for hdim in self._layers_dim[1:]:
            layers.append(torch.nn.Linear(dim, hdim))
            dim = hdim
        return layers

    def init_net(self, seed):
        torch.manual_seed(seed)
        for p in self.parameters():
            try:
                torch.nn.init.xavier_uniform_(p)
            except:
                torch.nn.init.constant_(p, 0)

    def get_activation(self, activation_i):
        name = self._activations[activation_i]
        if name == 'tanh':
            return torch.tanh
        elif name == 'relu':
            return torch.nn.functional.relu
        elif name == 'softplus':
            return torch.nn.functional.softplus
        else:
            raise Exception('unsupported activation function')

    def forward(self, x):
        for i, layer in enumerate(self._layers[: -1]):
            f = self.get_activation(i)
            x = f(layer(x))
        return self._layers[-1](x)

    def apply_method(self, x):
        return self.forward(x)

    def fit(self):
        pass

