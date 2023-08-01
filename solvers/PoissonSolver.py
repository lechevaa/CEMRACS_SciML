import numpy as np
import scipy

from typing import Dict


class PoissonSolver:
    def __init__(self, params: Dict):
        self._params = params
        self._equation = params['equation']
        # 1D domain should be: [a, b] with a > b
        self._domain = params['domain']
        # D is scalar for now
        self._D = params['D']
        self._nx = params['nx']
        self._x = np.linspace(self._domain[0], self._domain[1], self._nx)
        assert True

    @property
    def params(self):
        return self._params

    @property
    def equation(self):
        return self._equation

    @property
    def x(self):
        return self._x

    @property
    def D(self):
        return self._D

    @property
    def A(self):
        return self._A_assembly()

    @property
    def b(self):
        return self._F()

    def _A_assembly(self):
        nx = self._nx
        Ones = np.ones(nx - 2)

        d = np.hstack([1, -2 * Ones, 1])
        ds = np.hstack([0, Ones])
        di = np.hstack([Ones, 0])

        A_mat = scipy.sparse.diags([di, d, ds], [-1, 0, 1])

        # A_mat = -2 * np.eye(nx) + np.eye(nx, k=-1) + np.eye(nx, k=1)
        return A_mat

    def _F(self):
        a, b = self._domain
        nx = self._nx
        dx = (b - a) / (nx - 1)
        F = - (dx ** 2 / self._D) * np.ones(nx)
        F[0], F[-1] = 0., 0.
        return F

    def solve(self) -> np.ndarray:
        A = self._A_assembly().tocsc()
        b = self._F()
        U = scipy.sparse.linalg.spsolve(A, b)
        return U

    def update(self, params):
        self._params = params
        self._equation = params['equation']
        # 1D domain should be: [a, b] with a > b
        self._domain = params['domain']
        # D is scalar for now
        self._D = params['D']
        self._nx = params['nx']
        self._x = np.linspace(self._domain[0], self._domain[1], self._nx)

