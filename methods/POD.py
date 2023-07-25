from typing import Dict

from sklearn.decomposition import TruncatedSVD
import numpy as np

from solvers import Solver


class POD:
    def __init__(self, params: Dict):
        self._params = params
        # POD params are the args of the TruncatedSVD from scikit learn
        self._POD_params = params['POD_params']
        # U is the collection of data
        self._U = params['U']
        self._equation = params['equation']
        self._svd_model = self.svd()

    def svd(self):
        svd_model = TruncatedSVD(**self._POD_params)
        svd_model.fit(self._U)
        return svd_model

    def galerkin(self, V: np.ndarray, D):
        # new D is contained in params dict
        self._params['D'] = D
        solver = Solver(params=self._params)
        A = solver.A
        b = solver.b
        A_hat = V @ A @ V.T
        b_hat = V @ b
        alpha = np.linalg.solve(A_hat, b_hat)
        U_hat = V.T @ alpha
        return U_hat

    def apply_method(self, D):
        svd_model = self._svd_model
        V = svd_model.components_
        U_hat = self.galerkin(V, D)
        return U_hat

