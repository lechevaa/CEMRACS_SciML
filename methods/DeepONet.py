from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from methods.MLP import MLP

import copy
from tqdm import tqdm


class DeepONet(torch.nn.Module):
    def __init__(self, params):
        super(DeepONet, self).__init__()
        self._params = params
        self._solver_params = params['solver']
        self._method_params = params['method']
        self._losses = {'train': [], 'val': []}
        self._branch_params = params['method']['branch']
        self._trunk_params = params['method']['trunk']
        self.branch = MLP(params={'solver': params['solver'], 'method':  params['method']['branch']})
        self.trunk = MLP(params={'solver': params['solver'], 'method':  params['method']['trunk']})

        assert self.branch.device == self.trunk.device
        self._device = self.branch.device

    @property
    def loss_dict(self):
        return self._losses

    def forward(self, D, x):
        weights = self.branch(D)
        basis = self.trunk(x)
        return torch.matmul(weights, basis.T)

    def apply_method(self, D):
        domain = self._solver_params['domain']
        nx = self._solver_params['nx']
        x = torch.linspace(domain[0], domain[1], nx).view(-1, 1).to(self._device)

        if not torch.is_tensor(D):
            D = torch.Tensor(D)
        D = D.to(self._device)

        return self.forward(D, x).detach().cpu().numpy()

    @staticmethod
    def MSE(pred, true=0):
        return torch.square(true - pred).mean()

    def fit(self, hyperparameters: dict, D_train, D_val, U_train, U_val):

        epochs = hyperparameters['epochs']
        lr = hyperparameters['lr']
        optim_name = hyperparameters['optimizer']
        optimizer = getattr(optim, optim_name)(self.parameters(), lr=lr)
        X = torch.linspace(0., 1., self._solver_params['nx']).view(-1, 1)
        D_train = torch.Tensor(D_train).to(self._device)
        X = X.to(self._device)
        U_train = torch.Tensor(U_train).to(self._device)

        D_val = torch.Tensor(D_val).to(self._device)
        U_val = torch.Tensor(U_val).to(self._device)

        best_model = copy.deepcopy(self.state_dict())

        loading_bar = tqdm(range(epochs), colour='blue')
        for epoch in loading_bar:

            # Training of the model
            self.train()

            U_pred = self.forward(D_train, X)
            loss_tr = self.MSE(U_pred, U_train)

            loss_tr.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Validation of the model.
            self.eval()

            with torch.no_grad():
                U_pred = self.forward(D_val, X)
                loss_val = self.MSE(U_pred, U_val)

            self._losses['train'].append(loss_tr.item())
            self._losses['val'].append(loss_val.item())

            loading_bar.set_description('[tr : %.1e, val : %.1e]' % (loss_tr, loss_val))

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
        ax.legend()
        return

    def parity_plot(self, U, DX, ax, label, color):

        if torch.is_tensor(U):
            U = U.detach().numpy()
        U_true_norms = np.linalg.norm(U, 2, axis=1)

        U_pred = self.apply_method(DX[np.sort(np.unique(DX, return_index=True)[1])])
        U_pred_norms = np.linalg.norm(U_pred, 2, axis=1)
        ax.scatter(U_true_norms, U_pred_norms, s=10, label=label, color=color)
        ax.plot(U_true_norms, U_true_norms, 'r--', alpha=.5)

        ax.set_ylabel('$\|\widehat{\mathbf{u}}_D\|_2$', fontsize=18, labelpad=15)
        ax.set_xlabel('$\|\mathbf{u}_D\|_2$', fontsize=18, labelpad=15)
        return ax

    def load_loss_dict(self, loss_dict: Dict):
        self._losses = loss_dict
