from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from numpy.linalg import LinAlgError
from time import time
from oracles import lasso_duality_gap
from oracles import BarrierOracle
from datetime import datetime
from scipy.linalg import cho_factor, cho_solve


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5, 
                         tolerance_inner=1e-8, max_iter=100, 
                         max_iter_inner=20, t_0=1, gamma=10, 
                         c1=1e-4, lassodualitygap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    u_k = np.copy(u_0)
    t_k = t_0

    if display:
        print('Optimization debug information')
    times_0 = time()

    ldg_func = lassodualitygap if lassodualitygap is not None else lasso_duality_gap 
    print(A.shape, x_0.shape, b.shape)
    Ax_b = lambda y: A @ y - b
    ATAx_b = lambda y: A.T @ (A @ y - b) 
    ldg = lambda a: ldg_func(a, Ax_b(a), ATAx_b(a), b, reg_coef)
    line_search_options = {'method': 'Armijo', 'c1': c1}
    oracle = BarrierOracle(A, b, reg_coef, t_0)
    ldg_k = ldg(x_k)
    times_k = time() - times_0
    if trace:
        history['func'].append([oracle.antider_func(np.concatenate([x_k, u_k]))])
        history['time'].append([times_k])
        history['duality_gap'].append([ldg_k])
        if x_k.size <= 2:
            history['x'].append([x_k])

    for _ in range(max_iter):

        if ldg_k < tolerance:
            return (x_k, u_k), 'success', history
        
        oracle.t = t_k
        x_newton, mes_newton, _ = newton(oracle, np.concatenate([x_k, u_k]), 
                                         tolerance_inner, max_iter_inner, 
                                         line_search_options)
        x_k, u_k = np.array_split(x_newton, 2)
        if mes_newton == 'computational_error':
            return (x_k, u_k), 'computational_error', history

        t_k *= gamma
        ldg_k = ldg(x_k)
        times_k = time() - times_0
        if trace:
            history['func'].append([oracle.antider_func(np.concatenate([x_k, u_k]))])
            history['time'].append([times_k])
            history['duality_gap'].append([ldg_k])
            if x_k.size <= 2:
                history['x'].append([x_k])

        
    if ldg_k < tolerance:
        return (x_k, u_k), 'success', history
    
    return (x_k, u_k), 'iterations_exceeded', history


def optimal_alpha(vecs: np.ndarray, grads: np.ndarray):
    """
    Compute minimum value of alpha for linear search
    """

    x, u = np.split(vecs, 2)
    grad_x, grad_u = np.split(grads, 2)

    als = np.array([1.])
    theta = 0.99

    mask1 = grad_x > grad_u
    mask2 = grad_x < -grad_u

    als = np.append(als, theta * (u[mask1] - x[mask1]) / (grad_x[mask1] - grad_u[mask1]))
    als = np.append(als, theta * (x[mask2] + u[mask2]) / (-grad_x[mask2] - grad_u[mask2]))

    return np.min(als)


def newton(oracle, x_0, tolerance=1e-5, max_iter=100, line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    if display:
        print('Optimization debug information')

    times_0 = time()

    grad_k = lambda t: oracle.grad(t)
    func_k = lambda t: oracle.func(t)
    grad_0 = grad_k(x_0)
    times_k = time() - times_0
    if trace:
      history['time'] = [times_k]
      history['func'] = [func_k(x_k)]
      history['grad_norm'] = [np.linalg.norm(grad_0)]
      if x_k.size <= 2:
        history['x'] = [x_k]

    alpha0 = 1
    for _ in range(max_iter):
        if np.linalg.norm(grad_k(x_k)) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
            return x_k, 'success', history
        try:
            grad = grad_k(x_k)
            hess = oracle.hess(x_k)
            d_k = cho_solve((cho_factor(hess)), -grad)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        if not (np.all(np.isfinite(x_k)) and np.all(np.isfinite(d_k))):
            return x_k, 'computational_error', history

        alpha = line_search_tool.line_search(oracle=oracle, x_k=x_k, d_k=d_k, previous_alpha=optimal_alpha(x_k, d_k))
        x_k = x_k + alpha * d_k
        times = time() - times_0
        if trace:
          history['time'].append(times)
          history['func'].append(func_k(x_k))
          history['grad_norm'].append(np.linalg.norm(grad_k(x_k)))
          if x_k.size <= 2:
              history['x'].append(x_k)

        if np.linalg.norm(grad_k(x_k)) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history