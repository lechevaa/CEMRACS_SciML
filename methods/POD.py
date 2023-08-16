from typing import Dict

import torch
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt

from solvers import Solver


class POD:
    def __init__(self, params: Dict):

        self._params = params
        self._solver_params = params['solver']
        self._method_params = params['method']
        self._svd_model = None

    @staticmethod
    def svd(hyperparameters):
        svd_model = TruncatedSVD(**hyperparameters)
        return svd_model

    def fit(self, hyperparameters, U):
        self._svd_model = self.svd(hyperparameters)
        self._svd_model.fit(U)

    def galerkin(self, V: np.ndarray, D=None, y=None):
        # new D is contained in params dict
        if D is not None:
            if torch.is_tensor(D):
                D = D.numpy()
            self._solver_params['D'] = D
        if y is not None:
            if torch.is_tensor(y):
                y = y.numpy()
            if len(y) == 1:
                y = y[0]
            self._solver_params['y'] = y

        solver = Solver(params=self._params)
        A = solver.A
        b = solver.b
        A_hat = V @ A @ V.T
        b_hat = V @ b
        alpha = np.linalg.solve(A_hat, b_hat)
        U_hat = V.T @ alpha
        return U_hat

    def apply_method(self, phi, D=None, Y=None):
        svd_model = self._svd_model
        V = svd_model.components_
        U_hat = self.galerkin(V=V, D=D, y=Y)
        return U_hat

    def plot(self, ax):
        ax.grid(True)
        ax.set_axisbelow(True)

        n_components = self._svd_model.components_.shape[0]
        x = np.arange(n_components) + 1
        evr = self._svd_model.explained_variance_ratio_
        # ax.hist(evr, bins=x, alpha=0.5, color="blue", align='left', )
        evr = np.sort(evr)[::-1]
        x = np.insert(x, 0, 0)
        evr = np.insert(evr, 0, 0)

        ax.plot(x, evr.cumsum(), c='blue', marker='o', lw=2, label='Cumulative explained variance', zorder=1)
        ax.scatter(x, evr, color="red", alpha=0.8, label='Explained variance', zorder=2)
        ax.set_xlim([-0.5, x[-1] + 1.5])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks(x)
        ax.set_xlabel('Number of modes')
        ax.set_ylabel('Explained variance')

        ax.legend()

    def parity_plot(self, U, phi, ax, label, color, D=None, Y=None):
        if torch.is_tensor(phi):
            phi = phi.detach().cpu().numpy()
        if torch.is_tensor(D):
            D = D.detach().cpu().numpy()
        if torch.is_tensor(Y):
            Y = Y.detach().cpu().numpy()

        U_pred = []
        for i, p in enumerate(phi):
            d, y = None, None
            if D is not None:
                d = D[i]
            if Y is not None:
                y = Y[i]
            U_pred.append(self.apply_method(phi=p, D=d, Y=y))

        if torch.is_tensor(U):
            U = U.detach().cpu().numpy()

        U_pred_norm = np.linalg.norm(U_pred, 2, axis=1)
        U_true_norm = np.linalg.norm(U, 2, axis=1)

        ax.plot(U_true_norm, U_true_norm, 'r--', alpha=.5)
        ax.scatter(U_true_norm, U_pred_norm, s=10, label=label, color=color)

        ax.set_ylabel('$\|\widehat{\mathbf{u}}_D\|_2$', fontsize=18, labelpad=15)
        ax.set_xlabel('$\|\mathbf{u}_D\|_2$', fontsize=18, labelpad=15)
        return ax

