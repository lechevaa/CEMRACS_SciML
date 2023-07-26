import copy
from typing import Dict

import numpy as np

from solvers.PoissonSolver import PoissonSolver


class Solver:
    def __init__(self, params: Dict):
        assert isinstance(params, Dict)
        self._solver_params = params['solver']

        assert isinstance(params['solver']['equation'], str)
        self._equation = params['solver']['equation']

        if self._equation == 'Poisson':
            self._solver = PoissonSolver(params=self._solver_params)
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

    @property
    def D(self) -> np.ndarray:
        return self._solver.D

    def solve(self):
        if self._solver:
            return self._solver.solve()

    def change_D(self, new_D):
        self._solver_params = copy.deepcopy(self._solver_params)
        self._solver_params['D'] = new_D
        self._solver.update(self._solver_params)


