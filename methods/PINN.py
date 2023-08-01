import copy

import numpy as np
import torch
import torch.optim as optim
from methods.methodsDataset.PINNDataset import PINNDataset
from typing import Dict
from methods.MLP import MLP


class PINN(torch.nn.Module):
    def __init__(self, params: Dict):
        super(PINN).__init__()
        self._params = params
        self._solver_params = params['solver']
        self._method_params = params['method']
        self._losses = {'train': [], 'val': []}
        self._model = MLP(params=params['method'])

    def forward(self, x):
        return

    def apply_method(self):
        pass

    def loss(self, D, x):
        def MSE(pred, true=0):
            return torch.square(true - pred).mean()
        Dx = torch.cat([D, x], dim=1)
        u = self(Dx)

        u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]

        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True, allow_unused=True)[0]

        res = D * u_xx + 1

        x0 = torch.zeros_like(D)
        x1 = torch.ones_like(D)
        u0 = self(torch.cat([D, x0], dim=1))
        u1 = self(torch.cat([D, x1], dim=1))

        return MSE(u0) + MSE(u1), MSE(res)

    def fit(self):
        pass

    def plot(self):
        pass

    def parity_plot(self):
        pass