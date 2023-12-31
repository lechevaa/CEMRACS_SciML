from typing import Dict

import torch

from methods.POD import POD
from methods.MLP import MLP
from methods.DeepONet import DeepONet
from methods.PINN import PINN
from methods.FNO import FNO1d


class DDMethod:
    def __init__(self, params: Dict):
        self._params = params
        self._params_solver: Dict = params['solver']
        self._params_method: Dict = params['method']
        self._method_name = params['method']['method_name']
        self._method = self._find_method()

    def apply_method(self, phi, D=None, y=None):
        model = None
        # TODO: Data homogeneization for all methods
        if self._method_name in ['POD', 'MLP', 'PINN', 'DEEPONET', 'FNO', 'MLPINN']:
            model = self._method.apply_method(phi=phi, D=D, Y=y)
        else:
            print(f"model not handled yet got {self._method_name}")
        return model

    def _find_method(self):
        if self._method_name == 'POD':
            method = POD(params=self._params)
        elif self._method_name == 'MLP':
            method = PINN(params=self._params)
        elif self._method_name == 'DEEPONET':
            method = DeepONet(params=self._params)
        elif self._method_name == 'PINN':
            method = PINN(params=self._params)
        elif self._method_name == 'MLPINN':
            method = PINN(params=self._params)
        elif self._method_name == 'FNO':
            method = FNO1d(params=self._params)
        else:
            method = None
        return method

    def plot(self, ax):
        return self._method.plot(ax)

    def parity_plot(self, U, phi, ax, label, color, D=None, Y=None):
        return self._method.parity_plot(U, phi, ax, label, color, D=D, Y=Y)

    def fit(self, **args):
        print(f'Fitting {self._method_name}')
        if self._method_name in ['POD', 'MLP', 'PINN', 'DEEPONET', 'FNO', 'MLPINN']:
            self._method.fit(**args)
        print(f'{self._method_name} fitted')

    @property
    def state_dict(self):
        if self._method_name in ['MLP', 'PINN', 'DEEPONET', 'FNO', 'MLPINN']:
            return self._method.state_dict()

    @property
    def normalizers(self):
        if self._method_name in ['FNO']:
            return self._method.normalizers

    @property
    def method(self):
        return self._method

    def load_state_dict(self, path: str):
        if self._method_name in ['MLP', 'PINN', 'DEEPONET', 'FNO', 'MLPINN']:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            self._method.load_loss_dict(checkpoint['loss_dict'])
            self._method.load_state_dict(checkpoint['model_state_dict'])
            if 'normalizers' in checkpoint.keys():
                self._method.load_normalizers(checkpoint['normalizers'])

    def loss_dict(self):
        if self._method_name in ['MLP', 'PINN', 'DEEPONET', 'FNO', 'MLPINN']:
            return self._method.loss_dict
        else:
            return
