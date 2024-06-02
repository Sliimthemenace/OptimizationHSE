import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

   
class BarrierOracle(BaseSmoothOracle):
    
    def __init__(self, A: np.ndarray, b: np.ndarray, regcoef: float, t: float):
        self.A = A
        self.b = b

        self.regcoef = regcoef
        self.t = t

        self.matvec_Ax = lambda x: A @ x
        self.matvec_ATx = lambda x: A.T @ x

    def antider_func(self, pt: np.ndarray):
        x, u = np.array_split(pt, 2)
        return 1 / 2 * np.linalg.norm(self.matvec_Ax(x) - self.b) ** 2 + self.regcoef * np.sum(u)

    def func(self, pt: np.ndarray):
        x, u = np.array_split(pt, 2)
        up, um = np.log(u + x), np.log(u - x)
        return self.t * self.antider_func(pt) - np.sum(up + um)
    
    def grad(self, pt: np.ndarray):
        x, u = np.array_split(pt, 2)
        up, um = 1 / (u + x), 1 / (u - x)
        grad_func_x = self.t * self.matvec_ATx(self.matvec_Ax(x) - self.b) - up + um
        grad_func_u = self.t * self.regcoef * np.ones(len(u))
        return np.append(grad_func_x, grad_func_u)
    
    def hess(self, pt: np.ndarray):
        x, u = np.array_split(pt, 2)
        up, um = 1 / ((u + x) ** 2), 1 / ((u - x) ** 2)
        sm = up + um
        raz = up - um
        hess_func_u = np.diag(sm)
        hess_func_x = self.t * self.A.T @ self.A + np.diag(sm)
        hess_func_xu = np.diag(raz)
        return np.concatenate(
            (np.concatenate((hess_func_x, hess_func_xu), axis=1), 
             np.concatenate((hess_func_xu, hess_func_u), axis=1)), axis=0)


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    mu = np.min([1, regcoef / np.linalg.norm(ATAx_b, np.inf)]) * Ax_b
    nu = 1 / 2 * np.linalg.norm(Ax_b) ** 2 + regcoef * np.linalg.norm(x, 1) + 1 / 2 * np.linalg.norm(mu) ** 2 + b @ mu
    return nu