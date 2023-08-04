import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../methods/')

from methods.FNO import FNO1d
from solvers.Solver import Solver
import numpy as np
from sklearn.model_selection import train_test_split
import torch

nx = 101

params = {'solver': {'equation': 'Poisson', 'domain': [0, 1], 'D': 0., 'nx': nx},
          'method': {'layers_dim': [1, 10, 10, 10, 10], 'FourierLayers_modes': 3* [10], 'device': 'cuda', 'seed': 123}}

model = FNO1d(params=params)

Dmin, Dmax = 1, 10
D_list = D = np.linspace(Dmin, Dmax, 1000)
solver = Solver(params=params)
U_sols = []
for d in D_list:
    solver.change_D(new_D=d)
    U_sols.append(solver.solve())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
U_sols = np.stack(U_sols)
D = torch.Tensor(D).view(-1, 1).to(device)
D = D.repeat(1, nx).unsqueeze(-1)

U = torch.Tensor(U_sols).to(device)

D_train, D_val, U_train, U_val = train_test_split(D, U_sols, train_size=0.8)

hyperparameters = {'lr': 1e-3, 'epochs': 10000, 'optimizer': 'Adam', 'batch_size' : 64}

D_train = torch.Tensor(D_train).to(device)
U_train = torch.Tensor(U_train).to(device)

D_val = torch.Tensor(D_val).to(device)
U_val = torch.Tensor(U_val).to(device)

model.fit(hyperparameters=hyperparameters, D_train=D_train, D_val=D_val, U_train=U_train, U_val=U_val)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
model.plot(ax=ax)
plt.show()


pred = model(D_train).detach().cpu().squeeze(-1)
i = 0

x = torch.linspace(0, 1, nx)

plt.plot(x, U_train[i, :].detach().cpu(), lw = 2, label = 'true')
plt.plot(x, pred[i, :], 'r-.', label = 'fno')


plt.legend()
plt.show()
