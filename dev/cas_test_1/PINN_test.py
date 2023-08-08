import matplotlib.pyplot as plt

from methods.DataDrivenMethods import DDMethod
from solvers.Solver import Solver
import numpy as np
from sklearn.model_selection import train_test_split
import torch

params = {'solver': {'equation': 'Poisson', 'domain': [0, 1], 'D': 0., 'nx': 101},
          'method': {'layer_dims': [2, 100, 100, 100, 1], 'activations': 'tanh', 'device': 'cpu', 'seed': 123, 'method_name': 'PINN'}}

model = DDMethod(params=params)

Dmin, Dmax = 1, 10
D_list = D = np.linspace(Dmin, Dmax, 10)
solver = Solver(params=params)
U_sols = []
for d in D_list:
    solver.change_D(new_D=d)
    U_sols.append(solver.solve())

device = 'cpu'
U_sols = np.stack(U_sols)


d_train, d_val, u_train, u_val = train_test_split(D_list, U_sols, test_size=0.2, random_state=123)

D_train_repeated = torch.Tensor([[d] * len(solver.x) for d in d_train]).view(-1, 1)
D_val_repeated = torch.Tensor([[d] * len(solver.x) for d in d_val]).view(-1, 1)
x = torch.Tensor(solver.x).view(-1, 1)
X_train = x.repeat(d_train.shape[0], 1)
X_val = x.repeat(d_val.shape[0], 1)

DX_train = torch.cat((D_train_repeated, X_train), dim=1)
DX_val = torch.cat((D_val_repeated, X_val), dim=1)
dU_train = torch.Tensor(u_train.flatten())
dU_val = torch.Tensor(u_val.flatten())


hyperparameters = {'lr': 1e-3, 'epochs': 10000, 'optimizer': 'SGD', 'batch_size': 100}


model.fit(hyperparameters=hyperparameters, DX_train=DX_train, DX_val=DX_val)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# model

U_PINN = model.apply_method(1.)
solver.change_D(new_D=1.)
U_FD = solver.solve()

plt.plot(solver.x, U_FD, '--', label=f'FD: {np.linalg.norm(U_FD, 2):.2e}')
plt.plot(solver.x, U_PINN, label=f'PINN: {np.linalg.norm(U_PINN, 2):.2e}')
plt.legend()
plt.show()

