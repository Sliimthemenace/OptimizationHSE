import numpy as np
import matplotlib.pyplot as plt
from optimization import barrier_method_lasso

def size_n():
    np.random.seed(42)
    m = 200
    n_ax = [2 ** i for i in range(8)]
    regcoef = 1e-5

    # График зависимости от времени
    plt.figure(figsize=(10, 10))
    for n in n_ax:
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x_0 = np.zeros(n)
        u_0 = np.array([10] * n)
        _, _, history = barrier_method_lasso(A, b, regcoef, x_0, u_0, trace=True, max_iter=50)
        plt.plot(history['time'][:len(np.unique(history['duality_gap']))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'n={n}', linewidth=4, marker='v')
    plt.title('Зависимость от n')
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('log(duality_gap)', fontsize=14)
    plt.grid(True)
    plt.show()

    # График зависимости от номера итерации
    plt.figure(figsize=(10, 10))
    for n in n_ax:
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x_0 = np.zeros(n)
        u_0 = np.array([10] * n)
        _, _, history = barrier_method_lasso(A, b, regcoef, x_0, u_0, trace=True, max_iter=50)
        plt.plot([i for i in range(len(np.unique(history['duality_gap']) + 1))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'n={n}', linewidth=4, marker='.')
    plt.title('Зависимость от n')
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('log(duality_gap)', fontsize=14)
    plt.grid(True)
    plt.show()

def size_m():
    np.random.seed(42)
    n = 300
    m_ax = [10, 100, 300, 500]
    regcoef = 1e-5

    # График зависимости от времени
    plt.figure(figsize=(10, 10))
    for m in m_ax:
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x_0 = np.zeros(n)
        u_0 = np.array([10] * n)
        _, _, history = barrier_method_lasso(A, b, regcoef, x_0, u_0, trace=True, max_iter=50)
        plt.plot(history['time'][:len(np.unique(history['duality_gap']))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'm={m}', linewidth=4, marker='v')
    plt.title('Зависимость от m')
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('log(duality_gap)', fontsize=14)
    plt.grid(True)
    plt.show()

    # График зависимости от номера итерации
    plt.figure(figsize=(10, 10))
    for m in m_ax:
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x_0 = np.zeros(n)
        u_0 = np.array([10] * n)
        _, _, history = barrier_method_lasso(A, b, regcoef, x_0, u_0, trace=True, max_iter=50)
        plt.plot([i for i in range(len(np.unique(history['duality_gap']) + 1))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'm={m}', linewidth=4, marker='.')
    plt.title('Зависимость от m')
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('log(duality_gap)', fontsize=14)
    plt.grid(True)
    plt.show()

def coef_lambda():
    np.random.seed(100)
    n = 70
    m = 50
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    x_0 = np.array([5] * n)
    u_0 = np.array([10] * n)
    lambdas = [1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1]

    # График зависимости от времени
    plt.figure(figsize=(10, 10))
    for lambda_ in lambdas:
        _, _, history = barrier_method_lasso(A, b, lambda_, x_0, u_0, trace=True, max_iter=50)
        plt.plot(history['time'][:len(np.unique(history['duality_gap']))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'lambda={lambda_}', linewidth=4, marker='v')
    plt.title('Зависимость от lambda')
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('log(duality_gap)', fontsize=14)
    # plt.ylim(1, 1.8)
    # plt.xlim(-0.1, 2)
    plt.grid(True)
    plt.show()

    # График зависимости от номера итерации
    plt.figure(figsize=(10, 10))
    for lambda_ in lambdas:
        _, _, history = barrier_method_lasso(A, b, lambda_, x_0, u_0, trace=True, max_iter=50)
        plt.plot([i for i in range(len(np.unique(history['duality_gap']) + 1))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'lambda={lambda_}', linewidth=4, marker='.')
    plt.title('Зависимость от lambda')
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('log(duality_gap)', fontsize=14)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    size_n()
    size_m()
    coef_lambda()