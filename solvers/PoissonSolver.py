import numpy as np
import scipy

class PoissonSolver:
    
    def __init__(self, params: dict):
        
        self.params = params
        
        try :
            self.equation = params['equation']
        except :
            pass
        
        self.domain = params['domain']
        
        self.nx = params['nx']
        self.x = np.linspace(*self.domain, self.nx)
        
    def __A(self, D):
        
        # The approximation of D in the midpoints of the grid x
        D_ = (D[1:] + D[:-1])/2
        
        ds = np.hstack([0, D_[1:]])
        di = np.hstack([D_[:-1], 0])
        d = np.hstack([1, - (D_[1:] + D_[:-1]), 1])
        return scipy.sparse.diags([di, d, ds], [-1, 0, 1])
    
    @property
    def A(self) :
        return self.__A(self.params['D'])
    
    def __B(self, F):
        
        a, b = self.domain
        h = (b - a) / (self.nx - 1)
        B = - (h**2) *  F
        B[0], B[-1] = 0., 0.
        return B
    
    @property
    def B(self):
        return self.__B(self.params['F'])

    def __solve(self, A, B) -> np.ndarray:
        return scipy.sparse.linalg.spsolve(A.tocsc(), B)
    
    @property
    def solve(self) :
        return self.__solve(self.A, self.B)
    
    def Vsolve(self, vect: str, D = None, F = None) -> np.ndarray:
        
        U = []
        if vect == 'D':
            for d in D :
                A = self.__A(d)
                U.append(self.__solve(A, self.B))
            return np.stack(U)
        
        elif vect == 'F':
            for f in F :
                B = self.__B(f)
                U.append(self.__solve(self.A, B))
            return np.stack(U)
        
        elif vect == 'DF' and len(D) == len(F):
            for i, d in enumerate(D) :
                A = self.__A(d)
                B = self.__B(F[i])
                U.append(self.__solve(A, B))
            return np.stack(U)
        
        else:
            raise Exception(" vect must be either 'D', 'F' or 'DF' ")