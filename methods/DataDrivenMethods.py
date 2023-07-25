from typing import Dict
from methods.POD import POD


class DDMethod:
    def __init__(self, params: Dict):
        self._params = params
        self._method = params['method']

    def apply_method(self, D):
        if self._method == 'POD':
            method = POD(params=self._params)
            model = method.apply_method(D)
            return model
