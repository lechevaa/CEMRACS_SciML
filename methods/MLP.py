import copy

import torch
import torch.optim as optim

from typing import Dict


class MLP(torch.nn.Module):
    def __init__(self, params: Dict):
        self._params = params
        self._solver_params = params['solver']
        self._method_params = params['method']
        self._losses = {'train': [], 'val': []}

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

        # print(self)
        # print('device :', self.device)

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
        x = torch.Tensor([x]).to('cpu')
        return self.forward(x)

    def fit(self, hyperparameters: Dict, D_train, D_val, U_train, U_val):
        D_train = torch.Tensor(D_train).to(self.device)
        U_train = torch.Tensor(U_train).to(self.device)

        D_val = torch.Tensor(D_val).to(self.device)
        U_val = torch.Tensor(U_val).to(self.device)

        epochs = hyperparameters['epochs']
        lr = hyperparameters['lr']
        optim_name = hyperparameters['optimizer']
        optimizer = getattr(optim, optim_name)(self.parameters(), lr=lr)

        def loss_fn(x, y=0):
            return torch.square(y - x).mean()

        best_model = copy.deepcopy(self)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self(D_train)
            loss = loss_fn(output, U_train)
            loss.backward()
            optimizer.step()

            loss_train = loss.item()
            # Validation of the model.
            self.eval()
            with torch.no_grad():
                output = self(D_val)
                loss = loss_fn(output, U_val)

            loss_val = loss.item()
            self._losses['train'].append(loss_train)
            self._losses['val'].append(loss_val)

            # check if new best model
            if loss_val == min(self._losses['val']):
                best_model = copy.deepcopy(self)

        self.load_state_dict(best_model.state_dict())

    def plot_losses(self, ax):

        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12, labelpad=15)
        ax.plot(self._losses['train'], label='Training loss', alpha=.7)
        ax.plot(self._losses['val'], label='Validation loss', alpha=.7)

        ax.legend()
        return ax
