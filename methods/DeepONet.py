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
        u = torch.Tensor([u]).view(-1, 1).to('cpu')
        x_domain = torch.linspace(domain[0], domain[1], nx).view(-1, 1).to('cpu')
        u_x = []
        for x in x_domain:
            u_x.append(self.forward(u, x).detach())
        return np.array(u_x)

    def fit(self, hyperparameters: Dict, DX_train, DX_val, U_train, U_val):
        torch.manual_seed(self._branch_params['seed'])
        np.random.seed(self._branch_params['seed'])

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
                dx, u = data
                dx, u = dx.to(device), u.to(device)

                optimizer.zero_grad()

                output = self(dx[:, 0:1], dx[:, 1:2])

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
                    dx, u = data
                    dx, u = dx.to(device), u.to(device)
                    output = self(dx[:, 0:1], dx[:, 1:2])
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
        ax.plot(self._losses['train'], label=f'Training loss: {min(self._losses["train"]):.2}', alpha=.7)
        ax.plot(self._losses['val'], label=f'Validation loss: {min(self._losses["val"]):.2}', alpha=.7)

        ax.legend()
        return

    def parity_plot(self, U, DX, ax, label):

        U_pred_norms = []
        nx = self._solver_params['nx']
        U = U.detach().cpu().numpy()
        U_true_norms = [np.linalg.norm(U[nx*i: nx*(i+1)], 2) for i in range(U.shape[0]//nx)]
        for d in np.unique(DX[:, 0:1]):
            U_pred_temp = self.apply_method(d)
            U_pred_norms.append(np.linalg.norm(U_pred_temp, 2))
        ax.scatter(U_true_norms, U_pred_norms, s=10, label=label)
        ax.plot(U_true_norms, U_true_norms, 'r--', alpha=.5)

        ax.set_ylabel('$\|\widehat{\mathbf{u}}_D\|_2$', fontsize=18, labelpad=15)
        ax.set_xlabel('$\|\mathbf{u}_D\|_2$', fontsize=18, labelpad=15)
        return ax
