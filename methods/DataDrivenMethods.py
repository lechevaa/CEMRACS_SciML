from typing import Dict

import torch

from methods.POD import POD
from methods.MLP import MLP
from methods.DeepONet import DeepONet


class DDMethod:
    def __init__(self, params: Dict):
        self._params = params
        self._params_solver: Dict = params['solver']
        self._params_method: Dict = params['method']
        self._method_name = params['method']['method_name']
        self._method = self._find_method()

    def apply_method(self, D):
        model = self._method.apply_method(D)
        return model

    def _find_method(self):
        if self._method_name == 'POD':
            method = POD(params=self._params)
        elif self._method_name == 'MLP':
            method = MLP(params=self._params)
        elif self._method_name == 'DEEPONET':
            method = DeepONet(params=self._params)
        else:
            method = None
        return method

    def plot(self, ax):
        return self._method.plot(ax)

    def parity_plot(self, U, D, ax, label):
        return self._method.parity_plot(U, D, ax, label)

    def fit(self, **args):
        print(f'Fitting {self._method_name}')
        self._method.fit(**args)
        print(f'{self._method_name} fitted')

    def state_dict(self):
        if self._method_name in ['MLP', 'PINN', 'DEEPONET', 'FNO']:
            return self._method.state_dict()

    def load_state_dict(self, path: str):
        if self._method_name in ['MLP', 'PINN', 'DEEPONET', 'FNO']:
            self._method.load_state_dict(torch.load(path))

