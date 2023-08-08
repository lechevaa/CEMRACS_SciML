import torch
import torch.optim as optim

from methods.MLP import MLP

import copy
from tqdm import tqdm

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

        assert self.branch.device == self.trunk.device
        self._device = self.branch.device

    @property
    def loss_dict(self):
        return self._losses

    def forward(self, D, x):
        weights = self.branch(D)
        basis = self.trunk(x)
        return torch.matmul(weights, basis.T)

    def apply_method(self, D):
        domain = self._solver_params['domain']
        nx = self._solver_params['nx']
        x = torch.linspace(domain[0], domain[1], nx).view(-1, 1).to(self._device)
        D = torch.Tensor(D).view(-1, 1).to(self._device)

        return self.forward(D, x).detach().cpu().numpy()

    @staticmethod
    def MSE(pred, true=0):
        return torch.square(true - pred).mean()

    def fit(self, hyperparameters: dict, train_data, val_data):

        epochs = hyperparameters['epochs']
        lr = hyperparameters['lr']
        optim_name = hyperparameters['optimizer']
        optimizer = getattr(optim, optim_name)(self.parameters(), lr=lr)

        D_train, X_train, U_train = train_data
        D_train = torch.Tensor(D_train).to(self._device)
        X_train = torch.Tensor(X_train).to(self._device)
        U_train = torch.Tensor(U_train).to(self._device)

        D_val, X_val, U_val = val_data
        D_val = torch.Tensor(D_val).to(self._device)
        X_val = torch.Tensor(X_val).to(self._device)
        U_val = torch.Tensor(U_val).to(self._device)


        best_model = copy.deepcopy(self)

        loading_bar = tqdm(range(epochs), colour='blue')
        for epoch in loading_bar:

            self.train()

            U_pred = self.forward(D_train, X_train)
            loss_tr = self.MSE(U_pred, U_train)

            loss_tr.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Validation of the model.
            self.eval()

            with torch.no_grad():
                U_pred = self.forward(D_val, X_val)
                loss_val = self.MSE(U_pred, U_val)

            self._losses['train'].append(loss_tr.item())
            self._losses['val'].append(loss_val.item())

            loading_bar.set_description('[tr : %.1e, val : %.1e]' % (loss_tr, loss_val))

            # check if new best model
            if loss_val == min(self._losses['val']):
                best_model = copy.deepcopy(self)



        #self.load_state_dict(best_model.state_dict())

    def plot(self, ax):

        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12, labelpad=15)
        ax.set_ylabel('Loss', fontsize=12, labelpad=15)
        ax.plot(self._losses['train'], label=f'Training loss', alpha=.7)
        ax.plot(self._losses['val'], label=f'Validation loss', alpha=.7)

        ax.legend()
        return

    """
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
    """

    """
    def load_loss_dict(self, loss_dict: Dict):
        self._losses = loss_dict
    """
