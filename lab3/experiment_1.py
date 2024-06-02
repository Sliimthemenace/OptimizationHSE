import numpy as np
import matplotlib.pyplot as plt
from optimization import barrier_method_lasso

np.random.seed(42)
m = 100
n = 70
A = np.random.randn(m, n)
b = np.random.randn(m)
regcoef = 1e-5

x_0 = np.zeros(n)
u_0 = np.array([10] * n)

gamma = [2, 8, 32, 128, 256, 512]

# График зависимости от времени
plt.figure(figsize=(10, 10))
for gm in gamma:
    _, _, history = barrier_method_lasso(A, b, regcoef, x_0, u_0, gamma=gm, trace=True, max_iter=50)
    plt.plot(history['time'][:len(np.unique(history['duality_gap']))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'gamma={gm}', linewidth=4, marker='v')
plt.title('Зависимость от gamma')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('log(duality_gap)', fontsize=14)
plt.ylim(0.6, 1.8)
plt.xlim(-0.2, 3)
plt.grid(True)
plt.show()

# График зависимости от номера итерации
plt.figure(figsize=(10, 10))
for gm in gamma:
    _, _, history = barrier_method_lasso(A, b, regcoef, x_0, u_0, gamma=gm, trace=True, max_iter=50)
    plt.plot([i for i in range(len(np.unique(history['duality_gap']) + 1))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'gamma={gm}', linewidth=4, marker='v')
plt.title('Зависимость от gamma')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('log(duality_gap)', fontsize=14)
plt.ylim(0.8, 1.8)
plt.xlim(-0.2, 20)
plt.grid(True)
plt.show()


eps = [1e-8, 1e-7, 1e-6, 1e-4, 1e-3, 1e-2]

# График зависимости от времени
plt.figure(figsize=(10, 10))
for ep in eps:
    _, _, history = barrier_method_lasso(A, b, regcoef, x_0, u_0, tolerance_inner=ep, trace=True, max_iter=50)
    plt.plot(history['time'][:len(np.unique(history['duality_gap']))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'eps={ep}', linewidth=4, marker='.')
plt.title('Зависимость от eps')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('log(duality_gap)', fontsize=14)
plt.ylim(1, 1.8)
plt.grid(True)
plt.show()

# График зависимости от номера итерации
plt.figure(figsize=(10, 10))
for ep in eps:
    _, _, history = barrier_method_lasso(A, b, regcoef, x_0, u_0, tolerance_inner=ep, trace=True, max_iter=50)
    plt.plot([i for i in range(len(np.unique(history['duality_gap']) + 1))], np.log10(np.unique(history['duality_gap'])[::-1]), label=f'eps={ep}', linewidth=4, marker='.')
plt.title('Зависимость от eps')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('log(duality_gap)', fontsize=14)
plt.grid(True)
plt.show()