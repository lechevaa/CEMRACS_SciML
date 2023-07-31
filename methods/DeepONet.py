import copy

import numpy as np
import torch
import torch.optim as optim

from typing import Dict

from methods.MLP import MLP
from methods.methodsDataset.DeepONetDataset import DeepONetDataset


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

    def forward(self, u, y):
        weights = self.branch(u)
        basis = self.trunk(y)
        output = torch.matmul(weights, basis.T)
        return output

    def apply_method(self, u):
        domain = self._solver_params['domain']
        nx = self._solver_params['nx']
        u = torch.Tensor(u).view(-1, 1).to('cpu')
        x_domain = torch.linspace(domain[0], domain[1], nx).view(-1, 1).to('cpu')
        u_x = []
        for x in x_domain:
            u_x.append(self.forward(u, x).detach())
        return np.array(u_x)

    def fit(self, hyperparameters: Dict, DX_train, DX_val, U_train, U_val):
        DX_train = torch.Tensor(DX_train)
        U_train = torch.Tensor(U_train)

        DX_val = torch.Tensor(DX_val)
        U_val = torch.Tensor(U_val)

        trainDataset = DeepONetDataset(x=DX_train, y=U_train)
        valDataset = DeepONetDataset(x=DX_val, y=U_val)

        batch_size = hyperparameters['batch_size']

        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=False)

        epochs = hyperparameters['epochs']
        device = hyperparameters['device']
        lr = hyperparameters['lr']
        optim_name = hyperparameters['optimizer']
        optimizer = getattr(optim, optim_name)(self.parameters(), lr=lr)

        def loss_fn(x, y=0):
            return torch.square(y - x).mean()

        best_model = copy.deepcopy(self)
        for epoch in range(epochs):
            self.train()
            loss_train = 0.

            for i, data in enumerate(trainLoader):
                xd, u = data
                xd, u = xd.to(device), u.to(device)

                optimizer.zero_grad()

                output = self(xd[:, 0:1], xd[:, 1:2])

                loss = loss_fn(output, u)

                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            loss_train /= (i + 1)

            # Validation of the model.
            self.eval()
            loss_val = 0.
            with torch.no_grad():
                for i, data in enumerate(valLoader):
                    xd, u = data
                    xd, u = xd.to(device), u.to(device)
                    output = self(xd[:, 0:1], xd[:, 1:2])
                    loss_val += loss_fn(output, u).item()

            loss_val /= (i + 1)

            self._losses['train'].append(loss_train)
            self._losses['val'].append(loss_val)

            # check if new best model
            if loss_val == min(self._losses['val']):
                best_model = copy.deepcopy(self)

        self.load_state_dict(best_model.state_dict())

    def plot(self, ax):

        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12, labelpad=15)
        ax.set_xlabel('MSE Loss', fontsize=12, labelpad=15)
        ax.plot(self._losses['train'], label='Training loss', alpha=.7)
        ax.plot(self._losses['val'], label='Validation loss', alpha=.7)

        ax.legend()
        return ax

    def parity_plot(self, U, XD, ax, label):
        XD = torch.Tensor(XD).cpu()
        x = XD[:, 0:1]
        d = XD[:, 1:2]
        U_pred = self(u=d, y=x).detach().cpu().numpy()
        U_true = U.detach().cpu().numpy()
        U_pred_norm = np.linalg.norm(U_pred, 2, axis=1)
        U_true_norm = np.linalg.norm(U_true, 2, axis=1)
        ax.scatter(U_true_norm, U_pred_norm, s=10, label=label)
        ax.plot(U_true_norm, U_true_norm, 'r--', alpha=.5)

        ax.set_ylabel('$\|\widehat{\mathbf{u}}_D\|_2$', fontsize=18, labelpad=15)
        ax.set_xlabel('$\|\mathbf{u}_D\|_2$', fontsize=18, labelpad=15)
        return ax
