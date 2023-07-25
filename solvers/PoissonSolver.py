from typing import Dict


class PoissonSolver:
    def __init__(self, params: Dict):
        self._params = params
        self._equation = params['equation']
        assert True

    @property
    def params(self):
        return self._params

    @property
    def equation(self):
        return self._equation

    def solve(self):
        pass

