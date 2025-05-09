# task6_error_budget.py

import numpy as np
import matplotlib.pyplot as plt
from task1_simulation_model import t, n, P0, H, R
from task2_discretization import Phi, Lambda, Ga, S
from task3_simulation import x_s
from task4_kalman_filter import kalman_filter

# Monte Carlo-simulering
def monte_carlo(N, Phi, Lambda, Ga, P0, H, R, mi, u, n):
    n_states = Phi.shape[0]
    nm = int(n / mi) + 1  # Antall målinger
    k_meas = np.arange(0, n, mi)  # Målingstidspunkter

    # Lagre resultater for hver bane
    X_bar = np.zeros((N, n_states, n))  # A priori estimater
    X_hat = np.zeros((N, n_states, nm))  # A posteriori estimater
    X_true = np.zeros((N, n_states, n))  # Sanne tilstander
    E_bar = np.zeros((N, n_states, n))  # A priori feil
    E_hat = np.zeros((N, n_states, nm))  # A posteriori feil

    for j in range(N):
        # Generer ny stokastisk bane
        x = np.zeros((n_states, n))
        x[:, 0] = np.random.multivariate_normal(np.zeros(n_states), P0)
        
        # Korrekt form for støyvektoren - må matche Ga's kolonner
        for k in range(n-1):
            # Generer støy for armaturstrømmen
            v_k = np.random.normal(0, np.sqrt(0.02))  # Q_hat = 0.02
            
            # Viktig: Ga er en 3x3 matrise i vår implementasjon, så v_vec må være en 3x1 vektor
            v_vec = np.zeros((3, 1))
            v_vec[2, 0] = v_k  # Støy påvirker kun tredje tilstand (armaturstrøm)
            
            # Utfør tilstandsoppdateringen
            x[:, k+1:k+2] = Phi @ x[:, k:k+1] + Lambda.reshape(3, 1) * u + Ga @ v_vec
        
        X_true[j] = x

        # Generer målinger
        z = np.zeros((1, nm))
        for i in range(nm):
            k = k_meas[i]
            if k < n:
                z[:, i] = H @ x[:, k].reshape(n_states, 1) + np.random.randn(1) * np.sqrt(R)

        # Kjør Kalman-filter
        x_bar, x_hat, _, _, update_times = kalman_filter(z, Phi, Lambda, Ga, P0, H, R, mi, u, n)
        X_bar[j] = x_bar
        
        # Håndter x_hat som kan ha ulik lengde
        for i, k in enumerate(update_times):
            if i < nm and k < n:
                X_hat[j, :, i] = x_hat[:, i]

        # Beregn feil
        E_bar[j] = x - x_bar
        for i, k in enumerate(update_times):
            if i < nm and k < n:
                E_hat[j, :, i] = x[:, k] - x_hat[:, i]

    # Beregn statistikk
    m_hat = np.mean(E_hat, axis=0)  # Gjennomsnittlig a posteriori feil
    P_hat_N = np.zeros((nm, n_states, n_states))
    for k in range(nm):
        for j in range(N):
            e = E_hat[j, :, k] - m_hat[:, k]
            P_hat_N[k] += np.outer(e, e)
        if N > 1:  # Unngå divisjon med null
            P_hat_N[k] /= (N - 1)
    
    s_hat_N = np.sqrt(np.maximum(0, P_hat_N[:, 1, 1]))  # Standardavvik for hastighet (unngå negative verdier)
    
    # Hent standardavvik fra Kalman-filteret (antatt at dette er implementert i din kalman_filter funksjon)
    _, _, _, s_hat, _ = kalman_filter(z, Phi, Lambda, Ga, P0, H, R, mi, u, n)
    
    return X_true, X_bar, X_hat, E_bar, E_hat, m_hat, s_hat_N, s_hat, update_times

# Kjør Monte Carlo for N=10 og N=100
u = 1.0
mi = 100
N_10 = 10
N_100 = 100

# Fix for matplotlib escape sequences i labeler
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

X_true_10, X_bar_10, X_hat_10, E_bar_10, E_hat_10, m_hat_10, s_hat_N_10, s_hat_10, update_times = monte_carlo(N_10, Phi, Lambda, Ga, P0, H, R, mi, u, n)
X_true_100, X_bar_100, X_hat_100, E_bar_100, E_hat_100, m_hat_100, s_hat_N_100, s_hat_100, update_times = monte_carlo(N_100, Phi, Lambda, Ga, P0, H, R, mi, u, n)

# Plotting
t_m = t[update_times]  # Målingstidspunkter

# Plott 1: N=10, hastighetsestimater
plt.figure(figsize=(12, 8))
for j in range(N_10):
    plt.plot(t, X_bar_10[j, 1, :], 'r--', alpha=0.3, label='A priori $\\bar{x}_2$' if j == 0 else '')
    plt.plot(t_m[:len(X_hat_10[j, 1, :])], X_hat_10[j, 1, :len(t_m)], 'g-', alpha=0.3, 
             label='A posteriori $\\hat{x}_2$' if j == 0 else '')
plt.xlabel('Tid (s)')
plt.ylabel('Hastighet (m/s)')
plt.title('Hastighetsestimater for $N=10$ baner')
plt.legend()
plt.grid(True)
plt.savefig('mc_plot1.png')
plt.show()

# Plott 2: N=10, hastighetsfeil
plt.figure(figsize=(12, 8))
for j in range(N_10):
    plt.plot(t, E_bar_10[j, 1, :], 'r--', alpha=0.3, label='A priori $x_2 - \\bar{x}_2$' if j == 0 else '')
    plt.plot(t_m[:len(E_hat_10[j, 1, :])], E_hat_10[j, 1, :len(t_m)], 'g-', alpha=0.3, 
             label='A posteriori $x_2 - \\hat{x}_2$' if j == 0 else '')
plt.xlabel('Tid (s)')
plt.ylabel('Feil (m/s)')
plt.title('Hastighetsfeil for $N=10$ baner')
plt.legend()
plt.grid(True)
plt.savefig('mc_plot2.png')
plt.show()

# Plott 3: N=10, statistikk
plt.figure(figsize=(12, 8))
plt.plot(t_m[:len(m_hat_10[1, :])], m_hat_10[1, :len(t_m)], label='$\\hat{m}_2^N$ (Gjennomsnittlig feil)')
plt.plot(t_m[:len(s_hat_N_10)], s_hat_N_10[:len(t_m)], label='$\\hat{s}_2^N$ (Monte Carlo std)')
plt.plot(t_m[:len(s_hat_10)], s_hat_10[:len(t_m)], 'k--', label='$\\hat{s}_2$ (Kalman std)')
plt.xlabel('Tid (s)')
plt.ylabel('Statistikk')
plt.title('Statistikk for $N=10$')
plt.legend()
plt.grid(True)
plt.savefig('mc_plot3.png')
plt.show()

# Plott 4: N=100, statistikk
plt.figure(figsize=(12, 8))
plt.plot(t_m[:len(m_hat_100[1, :])], m_hat_100[1, :len(t_m)], label='$\\hat{m}_2^N$ (Gjennomsnittlig feil)')
plt.plot(t_m[:len(s_hat_N_100)], s_hat_N_100[:len(t_m)], label='$\\hat{s}_2^N$ (Monte Carlo std)')
plt.plot(t_m[:len(s_hat_100)], s_hat_100[:len(t_m)], 'k--', label='$\\hat{s}_2$ (Kalman std)')
plt.xlabel('Tid (s)')
plt.ylabel('Statistikk')
plt.title('Statistikk for $N=100$')
plt.legend()
plt.grid(True)
plt.savefig('mc_plot4.png')
plt.show()
