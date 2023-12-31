import copy
from typing import Dict

from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim

from methods.methodsDataset.FNODataset import FNODataset


class FourierLayer1d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, modes, seed=123):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        #  The kernel parametrization
        self.kernel = torch.nn.Parameter(
            (1 / (in_channels * out_channels)) * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

        self.Linear = torch.nn.Conv1d(in_channels, out_channels, 1)

    def cmul(self, x, w):
        """
        x : Tensor of size (i, j, k)
        w : Tensor of size (j, l, k)
        
        return a Tensor of size (i, l, k) which present the product of x and w along the axis j.
        """
        return torch.einsum("ijk, jlk -> ilk", x, w)

    def forward(self, v):
        """
        v : Tensor of size (b, d, n) where b is the batch size, d is the dimension of the output of the function v and 
            n is the number of grid points where v is evaluated.
        """

        b, d, n = v.shape
        Nyquist_frequency = (n // 2) + 1

        # Compute the fast Fourier transform of x
        F = torch.fft.rfft(v)

        # Compute the product of the kernel parametrisation and the relevant Fourier modes of F. 
        P = torch.zeros(b, self.out_channels, Nyquist_frequency, device=v.device, dtype=torch.cfloat)
        P[:, :, : self.modes] = self.cmul(F[:, :, : self.modes], self.kernel)

        # Compute the inverse fast Fourier transform of product
        K = torch.fft.irfft(P, n=n)

        return self.Linear(v) + K


class FNO1d(torch.nn.Module):

    def __init__(self, params: dict):

        super(FNO1d, self).__init__()

        self._params = params
        self._solver_params = params['solver']
        self._method_params = params['method']
        self._losses = {'train': [], 'val': []}
        self._l2_losses = {'train': [], 'val': []}

        self.n_features = params['method']['layers_dim'][0]
        self.lifting_dim = params['method']['layers_dim'][1]
        self.projection_dim = params['method']['layers_dim'][-1]

        self.FourierLayers_dim = params['method']['layers_dim'][1:]
        self.FourierLayers_modes = params['method']['FourierLayers_modes']

        self.Lifting = torch.nn.Linear(self.n_features, self.lifting_dim)

        self.FourierLayers = self.creat_FourierLayers()

        self.Projection = torch.nn.Sequential(
            torch.nn.Linear(self.projection_dim, 2 * self.projection_dim),
            torch.nn.GELU(),
            torch.nn.Linear(2 * self.projection_dim, 1)
        )

        if 'seed' in params.keys():
            self.init_net(params['method']['seed'])
        else:
            self.init_net(123)

        self.activation = torch.nn.GELU()

        self.padding = 9

        if 'device' in params.keys():
            if params['method']['device'] == 'cpu':
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self._normalizers = None

    def creat_FourierLayers(self):
        layers_dim = self.FourierLayers_dim
        modes = self.FourierLayers_modes
        layers = torch.nn.ModuleList()
        dim = layers_dim[0]
        for i, hdim in enumerate(layers_dim[1:]):
            layer = FourierLayer1d(dim, hdim, modes[i])
            layers.append(layer)
            dim = hdim
        return layers

    def init_net(self, seed):
        torch.manual_seed(seed)

        for p in self.parameters():
            try:
                torch.nn.init.xavier_uniform_(p)
            except:
                torch.nn.init.constant_(p, 0)

    def forward(self, x):
        x = self.Lifting(x)
        x = x.permute(0, 2, 1)

        x = torch.nn.functional.pad(x, [0, self.padding])

        for idx, layer in enumerate(self.FourierLayers):
            x = self.activation(layer(x))

        x = x[..., : -self.padding]

        x = x.permute(0, 2, 1)
        x = self.Projection(x)

        return x

    def apply_method(self, phi, D=None, Y=None):
        if not torch.is_tensor(phi):
            phi = torch.Tensor(phi).to(self.device)
        pred = None

        if D is not None and Y is not None:
            assert phi.shape[2] == 2
            pred = self.forward(phi).unsqueeze(-1)
        elif D is not None:
            if not torch.is_tensor(D):
                D = torch.Tensor(D).to(self.device)
            if torch.equal(phi, D):
                nx = self._solver_params['nx']
                # single D value evaluation
                if len(phi) == 1:
                    phi = torch.Tensor(phi).view(-1, 1)
                    phi = phi.repeat(1, nx).unsqueeze(-1)
                # multiple D value evaluation (risky test)
                elif len(phi) == nx:
                    phi = torch.Tensor(phi).view(1, -1).unsqueeze(-1)
                # single + multiple D of size (Nd, Nx) evaluation
                elif phi.shape[1] == 1:
                    phi = torch.Tensor(phi).view(-1, 1)
                    phi = phi.repeat(1, nx).unsqueeze(-1)
                else:
                    phi = torch.Tensor(phi).unsqueeze(-1)
                pred = self.forward(phi)

        elif Y is not None:
            if not torch.is_tensor(Y):
                Y = torch.Tensor(Y).to(self.device)

            if torch.equal(phi, Y):
                domain = self._solver_params['domain']
                nx = self._solver_params['nx']
                x = torch.linspace(domain[0], domain[1], nx).to(self.device)
                phi = self._solver_params['source_term'](phi, x)
                # single element evaluation
                if len(list(phi.shape)) == 1:
                    pred = self.forward(phi.view(1, -1).unsqueeze(-1))
                # multiple element evaluation
                elif len(list(phi.shape)) == 2:
                    pred = self.forward(phi.unsqueeze(-1))

        return pred.detach().cpu().numpy()

    @staticmethod
    def l2_error(U, U_star):
        return torch.linalg.vector_norm(U - U_star) / torch.linalg.vector_norm(U_star)

    def fit(self, hyperparameters: dict, D_train, D_val, U_train, U_val):
        torch.manual_seed(self._method_params['seed'])
        np.random.seed(self._method_params['seed'])

        D_train = torch.Tensor(D_train)
        U_train = torch.Tensor(U_train)

        D_val = torch.Tensor(D_val)
        U_val = torch.Tensor(U_val)

        trainDataset = FNODataset(x=D_train, y=U_train)
        valDataset = FNODataset(x=D_val, y=U_val)

        batch_size = hyperparameters['batch_size']

        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=False)

        del D_val, U_val
        del D_train, U_train
        del trainDataset, valDataset

        epochs = hyperparameters['epochs']
        lr = hyperparameters['lr']
        optim_name = hyperparameters['optimizer']
        optimizer = getattr(optim, optim_name)(self.parameters(), lr=lr)

        def loss_fn(x, y):
            return torch.square(y - x).mean()

        best_model = copy.deepcopy(self.state_dict())

        loading_bar = tqdm(range(epochs), colour='blue')
        for epoch in loading_bar:

            self.train()
            loss_train = 0.
            err_l2_train = 0.

            for i, data in enumerate(trainLoader):
                inputs, label = data
                inputs = inputs.to(self.device)
                label = label.to(self.device)
                optimizer.zero_grad()
                output = self(inputs).squeeze(-1)

                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                err_l2_train += self.l2_error(output, label).item()

            loss_train /= (i + 1)
            err_l2_train /= (i + 1)

            # Validation of the model.
            loss_val = 0.
            err_l2_val = 0.

            self.eval()
            with torch.no_grad():
                for i, data in enumerate(valLoader):
                    inputs, label = data
                    inputs = inputs.to(self.device)
                    label = label.to(self.device)
                    output = self(inputs).squeeze(-1)
                    loss_val += loss_fn(output, label).item()
                    err_l2_val += self.l2_error(output, label).item()

            loss_val /= (i + 1)
            self._losses['train'].append(loss_train)
            self._losses['val'].append(loss_val)

            self._l2_losses['train'].append(err_l2_train)
            self._l2_losses['val'].append(err_l2_val)
            # check if new best model
            if loss_val == min(self._losses['val']):
                best_model = copy.deepcopy(self.state_dict())

            loading_bar.set_description('[tr : %.1e, val : %.1e]' % (loss_train, loss_val))

        self.load_state_dict(best_model)

    def num_parameters(self):
        num = 0
        for p in self.parameters():
            num += torch.numel(p)
        return num


    def plot(self, ax):

        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12, labelpad=15)
        ax.set_ylabel('Loss', fontsize=12, labelpad=15)
        ax.plot(self._losses['train'], label=f'Training loss: {min(self._losses["train"]):.2e}', alpha=.7)
        ax.plot(self._losses['val'], label=f'Validation loss: {min(self._losses["val"]):.2e}', alpha=.7)
        ax.tick_params(axis="y", direction="in", which="both")
        ax.tick_params(axis="x", direction="in")
        ax.legend()
        return ax

    def parity_plot(self, U, phi, ax, label, color, D=None, Y=None):

        U_pred = self.apply_method(phi=phi, D=D, Y=Y)
        if torch.is_tensor(U):
            U = U.detach().numpy()

        U_pred_norm = np.linalg.norm(U_pred, 2, axis=1)
        U_true_norm = np.linalg.norm(U, 2, axis=1)

        ax.scatter(U_true_norm, U_pred_norm, s=10, label=label, color=color)
        ax.plot(U_true_norm, U_true_norm, 'r--', alpha=.5)

        ax.set_ylabel('$\|\widehat{\mathbf{u}}_D\|_2$', fontsize=18, labelpad=15)
        ax.set_xlabel('$\|\mathbf{u}_D\|_2$', fontsize=18, labelpad=15)
        return ax

    @property
    def loss_dict(self):
        return self._losses

    def load_loss_dict(self, loss_dict: Dict):
        self._losses = loss_dict

    @property
    def normalizers(self):
        return self._normalizers

    def load_normalizers(self, normalizers):
        self._normalizers = normalizers

