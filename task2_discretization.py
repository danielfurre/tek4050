import numpy as np
from scipy.linalg import expm, cholesky
from task1_simulation_model import F, L, G, Q_hat, dt

def c2d_deterministic(F, L, dt):
    """Convert continuous deterministic system matrices to discrete."""
    n = F.shape[0]
    F_aug = np.block([[F, L], [np.zeros((1, n)), 0]])
    exp_F_aug = expm(F_aug * dt)
    Phi = exp_F_aug[:n, :n]
    Lambda = exp_F_aug[:n, n]
    return Phi, Lambda

def c2d_stochastic(F, G, Q_hat, dt):
    """Convert continuous stochastic system matrices to discrete."""
    n = F.shape[0]
    GQG = G * Q_hat * G.T
    F1 = np.block([[F, GQG], [np.zeros((n, n)), -F.T]])
    exp_F1 = expm(F1 * dt)
    Fi12 = exp_F1[:n, n:]
    Fi22 = exp_F1[n:, n:]
    S = Fi12 @ np.linalg.inv(Fi22)
    Ga = cholesky(S, lower=False)
    return Ga

def cp2dpS(F, G, Q_hat, dt):
    """Compute discrete process noise covariance matrix S."""
    n = F.shape[0]
    GQG = G * Q_hat * G.T
    F1 = np.block([[F, GQG], [np.zeros((n, n)), -F.T]])
    exp_F1 = expm(F1 * dt)
    Fi12 = exp_F1[:n, n:]
    Fi22 = exp_F1[n:, n:]
    S = Fi12 @ np.linalg.inv(Fi22)
    return S

# Compute discrete matrices
Phi, Lambda = c2d_deterministic(F, L, dt)
Ga = c2d_stochastic(F, G, Q_hat, dt)
S = cp2dpS(F, G, Q_hat, dt)

print(Phi, Lambda, Ga, S)