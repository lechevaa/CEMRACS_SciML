import torch
import torch.optim as optim

from methods.MLP import MLP
from tqdm import tqdm

from tqdm import tqdm
import copy

class PINN(torch.nn.Module):
    def __init__(self, params: dict):
        super(PINN, self).__init__()

        self._params = params
        self._solver_params = params['solver']
        self._method_params = params['method']

        self._losses = {'train': {'data': [], 'ic_bc': [], 'residual': []},
                        'val': []}
        self._model = MLP(params=params)
        self._device = self._model.device

    def forward(self, D, x):
        Dx = torch.cat([D, x], dim=1)
        return self._model(Dx)

    def apply_method(self, D):
        domain = self._solver_params['domain']
        nx = self._solver_params['nx']

        D = torch.Tensor(D).reshape(-1, 1)
        D_nn = D.repeat(1, nx).reshape(-1, 1).to(self._device)

        x = torch.linspace(domain[0], domain[1], nx).view(-1, 1).to('cpu')
        x_nn = x.repeat(len(D), 1).to(self._device)

        return self.forward(D_nn, x_nn).detach().cpu().numpy()

    @property
    def loss_dict(self):
        return self._losses

    @staticmethod
    def MSE(pred, true=0):
        return torch.square(true - pred).mean()


    def phys_loss(self, D, x):

        if not x.requires_grad:
            x.requires_grad = True

        u = self.forward(D, x)

        u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]

        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True, allow_unused=True)[0]
        res = D * u_xx + 1

        x0 = torch.zeros_like(D)
        x1 = torch.ones_like(D)

        u0 = self.forward(D, x0) - 0.
        u1 = self.forward(D, x1) - 0.

        return self.MSE(u0) + self.MSE(u1), self.MSE(res)


    def data_loss(self, D, x, u_ex):
        u_pred = self.forward(D, x)
        return self.MSE(u_pred, u_ex)

    def fit(self, hyperparameters: dict, DX_train, DX_val, U_train, U_val, data_ratio = 1., physics_ratio = 1.):

        del trainDataset, valDataset
        del DX_train, DX_val

        epochs = hyperparameters['epochs']
        lr = hyperparameters['lr']
        optimizer = getattr(optim, hyperparameters['optimizer'])(self.parameters(), lr=lr)


        DX_train = torch.Tensor(DX_train).to(self._device)
        U_train = torch.Tensor(U_train).to(self._device)

        DX_val = torch.Tensor(DX_val).to(self._device)
        U_val = torch.Tensor(U_val).to(self._device)

        best_model = copy.deepcopy(self._model.state_dict())

        loading_bar = tqdm(range(epochs), colour='blue')
        for epoch in loading_bar:


            self.train()

            d, x = DX_train[:, 0:1], DX_train[:, 1:2]

            # Data driven loss
            if data_ratio == 0.:
                l_d = torch.zeros(1, device=self._device)
            else:
                l_d = self.data_loss(d, x, U_train)

            # Physics informed loss
            if physics_ratio == 0.:
                l_b, l_r = torch.zeros(1, device=self._device), torch.zeros(1, device=self._device)
            else:
                l_b, l_r = self.phys_loss(d, x)

            # Total loss
            l_tot = data_ratio * l_d + physics_ratio * (l_b + l_r)
            l_tot.backward()
            optimizer.step()
            optimizer.zero_grad()


            self._losses['train']['data'].append(l_d.item())
            self._losses['train']['ic_bc'].append(l_b.item())
            self._losses['train']['residual'].append(l_r.item())

            # Validation of the model.
            self.eval()

            with torch.no_grad():
                U_val_pred = self._model(DX_val)
                l_val = self.MSE(U_val, U_val_pred)
                self._losses['val'].append(l_val.item())


            # check if new best model
            if l_val == min(self._losses['val']):
                best_model = copy.deepcopy(self._model.state_dict())
                
            loading_bar.set_description('[tr : %.1e, val : %.1e]' %(l_tot, l_val))
            
        self._model.load_state_dict(best_model)

    def plot(self, ax):
        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12, labelpad=15)
        ax.set_ylabel('Loss', fontsize=12, labelpad=15)

        ax.plot(self._losses['train']['data'],
                label=f'Data driven loss', alpha=.7)

        ax.plot(self._losses['train']['ic_bc'],
                label=f'Boundary conditions loss', alpha=.7)

        ax.plot(self._losses['train']['residual'],
                label=f'Residual loss', alpha=.7)

        ax.plot(self._losses['val'],
                label=f'Validation loss', alpha=.7)
        ax.legend()
        return

    """
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
    """