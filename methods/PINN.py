import copy

import numpy as np
import torch
import torch.optim as optim
from methods.methodsDataset.PINNDataset import PINNDataset
from methods.methodsDataset.MLPINNDataset import MLPINNDataset
from typing import Dict
from methods.MLP import MLP
from tqdm import tqdm


class PINN(torch.nn.Module):
    def __init__(self, params: Dict):
        super(PINN, self).__init__()
        self._params = params
        self._solver_params = params['solver']
        self._method_params = params['method']
        self._losses = {'train': {'residual': [], 'ic_bc': []}, 'val': {'residual': [], 'ic_bc': []}}
        self._model = MLP(params=params)

    def forward(self, Dx):
        return self._model(Dx)

    def apply_method(self, D):
        domain = self._solver_params['domain']
        nx = self._solver_params['nx']
        D = torch.Tensor([D]*nx).view(-1, 1).to('cpu')
        x_domain = torch.linspace(domain[0], domain[1], nx).view(-1, 1).to('cpu')
        u_x = self.forward(torch.cat([D, x_domain], dim=1)).detach().cpu().numpy()
        return u_x

    @property
    def loss_dict(self):
        return self._losses

    def loss(self, D, x):
        def MSE(pred, true=0):
            return torch.square(true - pred).mean()

        if not x.requires_grad:
            x.requires_grad = True

        u = self._model(torch.cat([D, x], dim=1))

        u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]

        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True, allow_unused=True)[0]
        res = D * u_xx + 1

        x0 = torch.zeros_like(D)
        x1 = torch.ones_like(D)
        u0 = self._model(torch.cat([D, x0], dim=1)) - 0.
        u1 = self._model(torch.cat([D, x1], dim=1)) - 0.

        return MSE(u0) + MSE(u1), MSE(res)

    def fit(self, hyperparameters: Dict, DX_train, DX_val):
        torch.manual_seed(self._method_params['seed'])
        np.random.seed(self._method_params['seed'])

        DX_train = torch.Tensor(DX_train)
        DX_val = torch.Tensor(DX_val)

        trainDataset = PINNDataset(x=DX_train)
        valDataset = PINNDataset(x=DX_val)

        batch_size = hyperparameters['batch_size']

        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=False)

        del trainDataset, valDataset
        del DX_train, DX_val

        epochs = hyperparameters['epochs']
        device = self._method_params['device']
        lr = hyperparameters['lr']
        optim_name = hyperparameters['optimizer']
        optimizer = getattr(optim, optim_name)(self.parameters(), lr=lr)

        loading_bar = tqdm(range(epochs + 1), colour='blue')

        best_model = copy.deepcopy(self._model.state_dict())
        i = None
        for epoch in loading_bar:
            loading_bar.set_description('[epoch: %d ' % epoch)
            self.train()
            lb_train, lr_train = 0., 0.
            for i, dx in enumerate(trainLoader):
                d, x = dx[:, 0:1], dx[:, 1:2]
                d, x = d.to(device), x.to(device)

                optimizer.zero_grad()
                lb, lr = self.loss(d, x)
                l_tot = lb + lr
                l_tot.backward()

                optimizer.step()
                lb_train += lb.item()
                lr_train += lr.item()

            lr_train /= (i + 1)
            lb_train /= (i + 1)
            self._losses['train']['residual'].append(lr_train)
            self._losses['train']['ic_bc'].append(lb_train)
            # Validation of the model.
            lr_val, lb_val = 0., 0.

            for i, dx in enumerate(valLoader):
                d, x = dx[:, 0:1], dx[:, 1:2]
                d, x = d.to(device), x.to(device)
                lb, lr = self.loss(d, x)
                lb_val += lb.item()
                lr_val += lr.item()

            lr_val /= (i + 1)
            lb_val /= (i + 1)

            self._losses['val']['residual'].append(lr_val)
            self._losses['val']['ic_bc'].append(lb_val)

            # check if new best model
            val_tot = [sum(x) for x in zip(self._losses['val']['ic_bc'], self._losses['val']['residual'])]
            if lr_val + lb_val == min(val_tot):
                best_model = copy.deepcopy(self._model.state_dict())

        self._model.load_state_dict(best_model)

    def fit_supervised(self, hyperparameters: Dict, DX_train, DX_val, U_train, U_val):
        self._losses['train']['data_driven'] = []
        self._losses['val']['data_driven'] = []

        torch.manual_seed(self._method_params['seed'])
        np.random.seed(self._method_params['seed'])

        DX_train = torch.Tensor(DX_train)
        DX_val = torch.Tensor(DX_val)

        U_train = torch.Tensor(U_train)
        U_val = torch.Tensor(U_val)

        DX_train.requires_grad = True
        DX_val.requires_grad = True

        trainDataset = MLPINNDataset(x=DX_train, y=U_train)
        valDataset = MLPINNDataset(x=DX_val, y=U_val)

        batch_size = hyperparameters['batch_size']

        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=False)

        epochs = hyperparameters['epochs']
        device = self._method_params['device']
        lr = hyperparameters['lr']
        optim_name = hyperparameters['optimizer']
        optimizer = getattr(optim, optim_name)(self.parameters(), lr=lr)

        best_model = copy.deepcopy(self)
        i = None

        def MSE(pred, true=0):
            return torch.square(true - pred).mean()

        loading_bar = tqdm(range(epochs + 1), colour='blue')
        for epoch in loading_bar:
            loading_bar.set_description('[epoch: %d ' % epoch)
            self.train()
            lb_train, lr_train, l_dd_train = 0., 0., 0.
            for i, data in enumerate(trainLoader):
                dx, label = data
                dx = dx.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                l_dd = MSE(pred=self(dx), true=label)
                lb, lr = self.loss(dx[:, 0:1], dx[:, 1:2])
                l_tot = lb + lr + l_dd
                l_tot.backward()

                optimizer.step()
                lb_train += lb.item()
                lr_train += lr.item()
                l_dd_train += l_dd.item()

            lr_train /= (i + 1)
            lb_train /= (i + 1)
            l_dd_train /= (i + 1)
            self._losses['train']['residual'].append(lr_train)
            self._losses['train']['ic_bc'].append(lb_train)
            self._losses['train']['data_driven'].append(l_dd_train)
            # Validation of the model.
            lr_val, lb_val, l_dd_val = 0., 0., 0.

            for i, data in enumerate(valLoader):
                dx, label = data
                dx = dx.to(device)
                label = label.to(device)
                # dx.requires_grad = True
                l_dd = MSE(pred=self(dx), true=label)
                lb, lr = self.loss(dx[:, 0:1], dx[:, 1:2])
                lb_val += lb.item()
                lr_val += lr.item()
                l_dd_val += l_dd.item()

            lr_val /= (i + 1)
            lb_val /= (i + 1)
            l_dd_val /= (i + 1)

            self._losses['val']['residual'].append(lr_val)
            self._losses['val']['ic_bc'].append(lb_val)
            self._losses['val']['data_driven'].append(l_dd_val)
            # check if new best model
            val_tot = [sum(x) for x in zip(self._losses['val']['ic_bc'],
                                           self._losses['val']['residual'],
                                           self._losses['val']['data_driven'])]
            if lr_val + lb_val + l_dd_val == min(val_tot):
                best_model = copy.deepcopy(self._model)

        self._model.load_state_dict(best_model.state_dict())

    def plot(self, ax):
        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12, labelpad=15)
        ax.set_ylabel('Loss', fontsize=12, labelpad=15)
        # ax.plot(self._losses['train']['residual'],
        #         label=f'Train residual loss: {min(self._losses["train"]["residual"]):.2}', alpha=.7)
        # ax.plot(self._losses['val']['residual'],
        #         label=f'Val residual loss: {min(self._losses["val"]["residual"]):.2}', alpha=.7)
        # ax.plot(self._losses['train']['ic_bc'],
        #         label=f'Train boundary conditions loss: {min(self._losses["train"]["ic_bc"]):.2}', alpha=.7)
        # ax.plot(self._losses['val']['ic_bc'],
        #         label=f'Val boundary conditions loss: {min(self._losses["val"]["ic_bc"]):.2}', alpha=.7)
        #
        train_tot = [sum(x) for x in zip(self._losses['train']['ic_bc'], self._losses['train']['residual'])]
        ax.plot(train_tot,
                label=f'Train total loss: '
                      f'{min(train_tot):.2e}',
                alpha=.7)
        val_tot = [sum(x) for x in zip(self._losses['val']['ic_bc'], self._losses['val']['residual'])]
        ax.plot(val_tot,
                label=f'Val total loss: '
                      f'{min(val_tot):.2e}',
                alpha=.7)
        ax.legend()
        return

    def parity_plot(self, U, DX, ax, label):
        U_pred_norms = []
        U = U.detach().cpu().numpy()
        DX = DX.detach().cpu().numpy()
        U_true_norms = np.linalg.norm(U, 2, axis=1)

        for dx in np.unique(DX[:, 0:1]):
            U_pred_temp = self.apply_method(dx)
            U_pred_norms.append(np.linalg.norm(U_pred_temp, 2))
        ax.scatter(U_true_norms, U_pred_norms, s=10, label=label)
        ax.plot(U_true_norms, U_true_norms, 'r--', alpha=.5)

        ax.set_ylabel('$\|\widehat{\mathbf{u}}_D\|_2$', fontsize=18, labelpad=15)
        ax.set_xlabel('$\|\mathbf{u}_D\|_2$', fontsize=18, labelpad=15)
        return ax

    def load_loss_dict(self, loss_dict: Dict):
        self._losses = loss_dict
