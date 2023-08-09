import numpy as np
import scipy

from typing import Dict


class PoissonSolver:
    def __init__(self, params: Dict):
        self._params = params
        self._equation = params['equation']
        # 1D domain should be: [a, b] with a > b
        self._domain = params['domain']
        self._nx = params['nx']

        # D is scalar for now
        self._D = params['D']
        self._x = np.linspace(self._domain[0], self._domain[1], self._nx)
        
        # source term init
        self._y = None
        self._source_term = self.init_source_term()
        self._fx = self.init_fx()

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
    def fx(self):
        return self._fx

    @property
    def D(self):
        return self._D

    @property
    def Y(self):
        return self._y

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
        return A_mat

    def _F(self):
        a, b = self._domain
        nx = self._nx
        dx = (b - a) / (nx - 1)
        F = - (dx ** 2 / self._D) * self._fx
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
        self._nx = params['nx']
        self._source_term = self.init_source_term()
        # D is scalar for now
        self._D = params['D']
        self._x = np.linspace(self._domain[0], self._domain[1], self._nx)
        self._fx = self.init_fx()

    def init_source_term(self):
        if 'source_term' in self._params.keys():
            return self._params['source_term']
        else:
            return np.ones_like

    def init_fx(self):
        if 'source_term' in self._params.keys() and 'y' in self._params.keys():
            self._y = self._params['y']
            return self._source_term(self._params['y'], self._x)
        else:
            return self._source_term(self._x)
