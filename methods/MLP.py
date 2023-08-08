import copy

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from methods.methodsDataset.MLPDataset import MLPDataset
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

        self._normalizers = None
        # print(self)
        # print('device :', self.device)

    @property
    def loss_dict(self):
        return self._losses

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
        torch.manual_seed(self._method_params['seed'])
        np.random.seed(self._method_params['seed'])

        D_train = torch.Tensor(D_train)
        U_train = torch.Tensor(U_train)

        D_val = torch.Tensor(D_val)
        U_val = torch.Tensor(U_val)

        trainDataset = MLPDataset(x=D_train, y=U_train)
        self._normalizers = trainDataset.get_normalizers()
        valDataset = MLPDataset(x=D_val, y=U_val, normalizers=self._normalizers)

        batch_size = hyperparameters['batch_size']

        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=False)

        del trainDataset, valDataset
        del D_val, U_val

        epochs = hyperparameters['epochs']
        lr = hyperparameters['lr']
        optim_name = hyperparameters['optimizer']
        optimizer = getattr(optim, optim_name)(self.parameters(), lr=lr)

        def loss_fn(x, y=0):
            return torch.square(y - x).mean()

        loading_bar = tqdm(range(epochs + 1), colour='blue')
        best_model = copy.deepcopy(self.state_dict())
        for epoch in loading_bar:
            loading_bar.set_description('[epoch: %d ' % epoch)
            self.train()
            loss_train = 0.

            for i, data in enumerate(trainLoader):
                inputs, label = data
                inputs = inputs.to(self.device)
                label = label.to(self.device)
                optimizer.zero_grad()
                output = self(inputs)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            loss_train /= (i+1)

            # Validation of the model.
            loss_val = 0.
            self.eval()
            with torch.no_grad():
                for i, data in enumerate(valLoader):
                    inputs, label = data
                    inputs = inputs.to(self.device)
                    label = label.to(self.device)
                    optimizer.zero_grad()
                    output = self(inputs)
                    loss_val += loss_fn(output, label).item()

            loss_val /= (i+1)
            self._losses['train'].append(loss_train)
            self._losses['val'].append(loss_val)

            # check if new best model
            if loss_val == min(self._losses['val']):
                best_model = copy.deepcopy(self.state_dict())

        self.load_state_dict(best_model)

    def plot(self, ax):

        ax.grid(True)
        ax.set_yscale('log')

        ax.set_xlabel('Epoch', fontsize=12, labelpad=15)
        ax.set_ylabel('Loss', fontsize=12, labelpad=15)
        ax.plot(self._losses['train'], label=f'Training loss: {min(self._losses["train"]):.2e}', alpha=.7)
        ax.plot(self._losses['val'], label=f'Validation loss: {min(self._losses["val"]):.2e}', alpha=.7)
        ax.tick_params(axis="y", direction="in", which="both")
        ax.tick_params(axis="x", direction="in")
        ax.legend()
        return ax

    def parity_plot(self, U, D, ax, label):
        xy_normalizer = self._normalizers
        D = torch.Tensor(D).cpu()
        if xy_normalizer:
            x_normalizer, y_normalizer = xy_normalizer
            if x_normalizer:
                D = x_normalizer.encode(D)
            U_pred = self(D).detach().cpu()
            if y_normalizer:
                U_pred = y_normalizer.decode(U_pred)
            U_pred.numpy()
        else:
            U_pred = self(D).detach().cpu().numpy()

        U_true = U.detach().cpu().numpy()
        U_pred_norm = np.linalg.norm(U_pred, 2, axis=1)
        U_true_norm = np.linalg.norm(U_true, 2, axis=1)
        ax.scatter(U_true_norm, U_pred_norm, s=10, label=label)
        ax.plot(U_true_norm, U_true_norm, 'r--', alpha=.5)

        ax.set_ylabel('$\|\widehat{\mathbf{u}}_D\|_2$', fontsize=18, labelpad=15)
        ax.set_xlabel('$\|\mathbf{u}_D\|_2$', fontsize=18, labelpad=15)
        return ax

    def load_loss_dict(self, loss_dict: Dict):
        self._losses = loss_dict

    @property
    def normalizers(self):
        return self._normalizers

    def load_normalizers(self, normalizers):
        self._normalizers = normalizers

