import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../../methods/')

from methods.FNO import FNO1d
from methods.DataDrivenMethods import DDMethod
from solvers.Solver import Solver
import numpy as np
from sklearn.model_selection import train_test_split
import torch


def delta(y, x, dy=1., dx=0.):
    """
    y : int, float or ndarray of size 1
    x : ndarray

    return dy if x = y and dx otherwise
    """
    return np.where(x == y, dy, dx)


nx = 101

params = {'solver': {'equation': 'Poisson', 'domain': [0, 1], 'D': 0., 'nx': nx,
                     'source_term': lambda x: delta(0.5, x)},
          'method': {'layers_dim': [1, 20, 20, 20, 20], 'FourierLayers_modes': 3*[12], 'device': 'cuda', 'seed': 123,
                     'method_name': 'FNO'}}

model = DDMethod(params=params)

Dmin, Dmax = 0.1, 10
D_list = D = np.linspace(Dmin, Dmax, 1000)
solver = Solver(params=params)
U_sols = []
for d in D_list:
    solver.change_D(new_D=d)
    U_sols.append(solver.solve())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
U_sols = np.stack(U_sols)
# D = torch.Tensor(D).view(-1, 1).to(device)
# D = D.repeat(1, nx).unsqueeze(-1)

U = torch.Tensor(U_sols).to(device)

D_train, D_val, U_train, U_val = train_test_split(D, U_sols, train_size=0.8, random_state=123)

hyperparameters = {'lr': 1e-3, 'epochs': 2000, 'optimizer': 'Adam', 'batch_size': 64}

D_train = torch.Tensor(D_train).to(device)
U_train = torch.Tensor(U_train).to(device)

D_val = torch.Tensor(D_val).to(device)
U_val = torch.Tensor(U_val).to(device)

model.fit(hyperparameters=hyperparameters, D_train=D_train, D_val=D_val, U_train=U_train, U_val=U_val)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
model.plot(ax=ax)
plt.show()


i = 0

x = torch.linspace(0, 1, nx)

figs, ax = plt.subplots(1, 1)
model.parity_plot(U=U_train, D=D_train, ax=ax, label='Train')
model.parity_plot(U=U_val, D=D_val, ax=ax, label='Val')

#
# plt.legend()
plt.show()

torch.save({'model_state_dict': model.state_dict,
            'loss_dict': model.loss_dict(),
            'normalizers': model.normalizers
            }, 'FNO.pt')
