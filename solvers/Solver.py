from typing import Dict
from solvers.PoissonSolver import PoissonSolver


class Solver:
    def __init__(self, params: Dict):
        self._params = params
        self._equation = params['equation']

        if self._equation == 'Poisson':
            self._solver = PoissonSolver(params=self._params)
        else:
            # Solver not implemented yet
            self._solver = None

    @property
    def equation(self) -> str:
        return self._equation

    def solve(self):
        if self._solver:
            self._solver.solve()
