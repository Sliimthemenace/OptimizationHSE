import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
from time import time


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    # TODO: Implement Conjugate Gradients method.
    if display:
        print('Optimization debug information')
    times_0 = time()

    g_k = matvec(x_k) - b
    times_k = time() - times_0
    if trace:
       history['time'] = [times_k]
       history['residual_norm'] = [np.linalg.norm(g_k)]
       if x_k.size <= 2:
          history['x'] = [x_k]

    d_k = -g_k
    max_iter = min(max_iter, 2 * len(x_k)) if max_iter else 2 * len(x_k)
    for _ in range(max_iter):
        if np.linalg.norm(g_k) <= tolerance * np.linalg.norm(b):
            return x_k, 'success', history
        
        alpha_k = (g_k.T @ g_k) / (d_k @ matvec(d_k))
        g_k_prev = np.copy(g_k)
        x_k = x_k + alpha_k * d_k
        g_k = g_k_prev + alpha_k * matvec(d_k)
        beta_k = (g_k.T @ g_k) / (g_k_prev.T @ g_k_prev)
        d_k = -g_k + beta_k * d_k
        times = time() - times_0
        if trace:
            history['time'].append(times)
            history['residual_norm'].append(np.linalg.norm(g_k))
            if x_k.size <= 2:
                history['x'].append(x_k)
    
    if np.linalg.norm(g_k) <= tolerance * np.linalg.norm(b):
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.
    if display:
        print('Optimization debug information')
    times_0 = time()

    grad_k = lambda t: oracle.grad(t)
    func_k = lambda t: oracle.func(t)
    grad_0 = grad_k(x_0)
    times_k = time() - times_0
    if trace:
        history['func'] = [func_k(x_k)]
        history['time'] = [times_k]
        history['grad_norm'] = [np.linalg.norm(grad_0)]
        if x_k.size <= 2:
            history['x'] = [x_k]

    def bfgs_multiply(v, H, gamma_0):
        if len(H) == 0:
            return gamma_0 * v
        
        s, y = H[-1]
        H_new = H[:-1]
        v_new = v - (s @ v) / (y @ s) * y
        z = bfgs_multiply(v_new, H_new, gamma_0)
        return z + (s @ v - y @ z) / (y @ s) * s

    def lbfgs_direction(H, grad):
        if len(H) == 0:
            return -grad
        
        s, y = H[-1]
        gamma_0 = (y @ s) / (y @ y)
        return bfgs_multiply(-grad, H, gamma_0)

    H = []
    for _ in range(max_iter):
        if np.linalg.norm(grad_k(x_k)) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
            return x_k, 'success', history
        d = lbfgs_direction(H, grad_k(x_k))
        alpha = line_search_tool.line_search(oracle=oracle, x_k=x_k, d_k=d)
        x_k1 = x_k + alpha * d
        grad_k1 = grad_k(x_k1)
        H.append((x_k1 - x_k, grad_k1 - grad_k(x_k)))
        if len(H) > memory_size:
            H = H[1:]
        x_k = x_k1
        times = time() - times_0
        if trace:
            history['time'].append(times)
            history['func'].append(func_k(x_k))
            history['grad_norm'].append(np.linalg.norm(grad_k1))
            if x_k.size <= 2:
                history['x'].append(x_k)

        if np.linalg.norm(grad_k(x_k)) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
            return x_k, 'success', history
        
    if np.linalg.norm(grad_k(x_k)) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
        return x_k, 'success', history
    return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
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
    d_k = -grad_0
    times_k = time() - times_0
    if trace:
        history['func'] = [func_k(x_k)]
        history['time'] = [times_k]
        history['grad_norm'] = [np.linalg.norm(grad_0)]
        if x_k.size <= 2:
            history['x'] = [x_k]

    alpha0 = 1
    grad = 0
    for _ in range(max_iter):
        if np.linalg.norm(grad_k(x_k)) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
            return x_k, 'success', history
        grad = grad_k(x_k)
        eps_k = min(0.5, np.sqrt(np.linalg.norm(grad)))
        while True:
            hess = lambda v: oracle.hess_vec(x_k, v)
            d_k, _, _ = conjugate_gradients(hess, -grad, d_k, eps_k)
            if grad @ d_k < 0:
                break
            else:
                eps_k /= 10
        alpha = line_search_tool.line_search(oracle=oracle, x_k=x_k, d_k=d_k, previous_alpha=alpha0)
        x_k = x_k + alpha * d_k
        grad = grad_k(x_k)
        times = time() - times_0
        if trace:
            history['time'].append(times)
            history['func'].append(func_k(x_k))
            history['grad_norm'].append(np.linalg.norm(grad))
            if x_k.size <= 2:
                history['x'].append(x_k)

        if np.linalg.norm(grad) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
            return x_k, 'success', history
        
    if np.linalg.norm(grad) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
        return x_k, 'success', history
    return x_k, 'iterations_exceeded', history