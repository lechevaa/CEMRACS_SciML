import matplotlib.pyplot as plt

from methods.MLP import MLP
from solvers.Solver import Solver
import numpy as np
from sklearn.model_selection import train_test_split
import torch

params = {'solver': {'equation': 'Poisson', 'domain': [0, 1], 'D': 0., 'nx': 101},
          'method': {'layer_dims': [1, 100, 100, 100, 101], 'activations': 'tanh', 'device': 'cpu', 'seed': 123}}

model = MLP(params=params)

Dmin, Dmax = 1, 10
D_list = D = np.linspace(Dmin, Dmax, 1000)
solver = Solver(params=params)
U_sols = []
for d in D_list:
    solver.change_D(new_D=d)
    U_sols.append(solver.solve())

device = 'cpu'
U_sols = np.stack(U_sols)
D = torch.Tensor(D).view(-1, 1).to(device)
U = torch.Tensor(U_sols).to(device)

D_train, D_val, U_train, U_val = train_test_split(D, U_sols, train_size=0.8)

hyperparameters = {'lr': 1e-3, 'epochs': 10000, 'optimizer': 'SGD'}

D_train = torch.Tensor(D_train).to(device)
U_train = torch.Tensor(U_train).to(device)

D_val = torch.Tensor(D_val).to(device)
U_val = torch.Tensor(U_val).to(device)

model.fit(hyperparameters=hyperparameters, D_train=D_train, D_val=D_val, U_train=U_train, U_val=U_val)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
model.plot_losses(ax=ax)
plt.show()
