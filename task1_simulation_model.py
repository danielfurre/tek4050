import numpy as np

# Simulation parameters
t0 = 0
tf = 100
dt = 0.01
n = int((tf - t0) / dt) + 1
t = np.linspace(t0, tf, n)

# Control input
u = 1.0  # Constant control input

# System parameters
T2 = 5.0  # s
T3 = 1.0  # s
Q_hat = 2 * (0.1)**2  # Process noise spectral density
R = 1.0  # Measurement noise variance
P0 = np.diag([1.0, 0.01, 0.01])  # Initial state covariance

# System matrices
F = np.array([[0, 1, 0],
              [0, -1/T2, 1/T2],
              [0, 0, -1/T3]])
L = np.array([[0],
              [0],
              [1/T3]])
G = np.array([[0],
              [0],
              [1]])
H = np.array([[1, 0, 0]])