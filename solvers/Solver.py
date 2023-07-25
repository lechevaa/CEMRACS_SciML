from typing import Dict

import numpy as np

from solvers.PoissonSolver import PoissonSolver


class Solver:
    def __init__(self, params: Dict):
        assert isinstance(params, Dict)
        self._params = params

        assert isinstance(params['equation'], str)
        self._equation = params['equation']

        if self._equation == 'Poisson':
            self._solver = PoissonSolver(params=self._params)
        else:
            # Solver not implemented yet
            self._solver = None
        assert True

    @property
    def equation(self) -> str:
        return self._equation

    @property
    def x(self) -> np.ndarray:
        return self._solver.x

    @property
    def A(self) -> np.ndarray:
        return self._solver.A

    @property
    def b(self) -> np.ndarray:
        return self._solver.b

    def solve(self):
        if self._solver:
            return self._solver.solve()

    def change_D(self, D):
        self._params['D'] = D
