import numpy as np
import matplotlib.pyplot as plt
from task1_simulation_model import t, n
from task2_discretization import Phi, Lambda, Ga

# Define control input
u = 1.0  # Constant control input as specified

# Deterministic simulation
def run_deterministic(Phi, Lambda, u, n):
    x = np.zeros((3, n))
    for k in range(n-1):
        x[:, k+1] = Phi @ x[:, k] + Lambda * u
    return x

# Stochastic simulation
def run_stochastic(Phi, Lambda, Ga, u, n):
    x = np.zeros((3, n))
    np.random.seed(0)  # For reproducibility
    for k in range(n-1):
        w_k = np.random.randn(3)
        x[:, k+1] = Phi @ x[:, k] + Lambda * u + Ga @ w_k
    return x

# Run simulations
x_d = run_deterministic(Phi, Lambda, u, n)
x_s = run_stochastic(Phi, Lambda, Ga, u, n)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, x_d[1, :], label='Deterministic $x_2$')
plt.plot(t, x_s[1, :], label='Stochastic $x_2$', alpha=0.7)
plt.plot(t, u * np.ones_like(t), label='$u$', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Velocity $x_2$')
plt.title('Deterministic vs Stochastic Simulation')
plt.legend()
plt.grid(True)
plt.savefig('task3_simulation.png')
plt.close()