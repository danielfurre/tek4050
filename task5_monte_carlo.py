# task6_error_budget.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# Importer nødvendige funksjoner fra dine tidligere filer
from task2_discretization import c2d_deterministic, c2d_stochastic, cp2dpS
from task4_kalman_filter import kalman_filter

def error_budget(F, G, H, Q_hat, R, P0, t0, tf, dt):
    """
    Kjør feilbudsjett for det optimale Kalman-filteret.
    
    Args:
        F: Systemmatrise
        G: Prosess-støymatrise
        H: Målingsmatrise
        Q_hat: Kontinuerlig prosesstøy-kovarians
        R: Målingsstøy-kovarians
        P0: Initial kovarians
        t0: Starttid
        tf: Sluttid
        dt: Tidsskritt
    
    Returns:
        time: Tidspunkter
        s_pos_process: Standardavvik for posisjon fra prosesstøy
        s_pos_meas: Standardavvik for posisjon fra målingsstøy
        s_pos_init: Standardavvik for posisjon fra initialtilstand
        s_vel_process: Standardavvik for hastighet fra prosesstøy
        s_vel_meas: Standardavvik for hastighet fra målingsstøy
        s_vel_init: Standardavvik for hastighet fra initialtilstand
        s_curr_process: Standardavvik for strøm fra prosesstøy
        s_curr_meas: Standardavvik for strøm fra målingsstøy
        s_curr_init: Standardavvik for strøm fra initialtilstand
        s_pos_total: Total standardavvik for posisjon
        s_vel_total: Total standardavvik for hastighet
        s_curr_total: Total standardavvik for strøm
        s_pos_kalman: Kalman-filter standardavvik for posisjon
    """
    # Tidsdimensjoner
    t = np.arange(t0, tf + dt, dt)
    n = len(t)
    
    # Diskretisere systemet
    Phi, Lambda = c2d_deterministic(F, np.zeros((3, 1)), dt)
    S = cp2dpS(F, G, Q_hat, dt)
    
    # Målingsintervall (1 Hz)
    mi = int(1 / dt)  # 100 tidssteg = 1 sekund
    
    # Initialiser feilkovarianser for hver kilde
    # Prosesstøy
    Pe_process = np.zeros((n, 3, 3))
    # Målingsstøy
    Pe_meas = np.zeros((n, 3, 3))
    # Initialtilstand
    Pe_init = np.zeros((n, 3, 3))
    # Total
    Pe_total = np.zeros((n, 3, 3))
    
    # Sett initialverdier
    Pe_init[0] = P0
    Pe_total[0] = P0
    
    # Kjør Kalman-filter for å få standardavvik fra filteret
    # Generere dummy-målinger (vi trenger bare standardavvik)
    nm = int(n / mi) + 1
    z = np.zeros((1, nm))
    x_bar, x_hat, P_bar, P_hat = kalman_filter(z, Phi, np.zeros((3, 1)), 
                                              np.zeros((3, 3)), P0, 
                                              H, R, mi, 0, n)
    
    # Tidssteg-løkke for å propagere feilkovariansene
    for k in range(n-1):
        # Tidoppdatering for hver kilde
        # Prosesstøy
        Pe_process[k+1] = Phi @ Pe_process[k] @ Phi.T + S
        
        # Initialtilstand
        Pe_init[k+1] = Phi @ Pe_init[k] @ Phi.T
        
        # Målingsstøy
        Pe_meas[k+1] = Phi @ Pe_meas[k] @ Phi.T
        
        # Målingsoppdatering ved hvert målingstidspunkt
        if (k+1) % mi == 0:
            # Beregn Kalman-forsterkning
            K = P_bar[k+1] @ H.T @ np.linalg.inv(H @ P_bar[k+1] @ H.T + R)
            
            # Oppdater feilkovarianser
            I_KH = np.eye(3) - K @ H
            
            # Prosesstøy
            Pe_process[k+1] = I_KH @ Pe_process[k+1] @ I_KH.T + K @ R @ K.T
            
            # Initialtilstand
            Pe_init[k+1] = I_KH @ Pe_init[k+1] @ I_KH.T
            
            # Målingsstøy
            Pe_meas[k+1] = I_KH @ Pe_meas[k+1] @ I_KH.T + K @ R @ K.T
    
    # Beregn total feilkovarians (summen av alle kilder)
    Pe_total = Pe_process + Pe_meas + Pe_init
    
    # Ekstraherer standardavvik
    s_pos_process = np.sqrt(Pe_process[:, 0, 0])
    s_pos_meas = np.sqrt(Pe_meas[:, 0, 0])
    s_pos_init = np.sqrt(Pe_init[:, 0, 0])
    s_pos_total = np.sqrt(Pe_total[:, 0, 0])
    
    s_vel_process = np.sqrt(Pe_process[:, 1, 1])
    s_vel_meas = np.sqrt(Pe_meas[:, 1, 1])
    s_vel_init = np.sqrt(Pe_init[:, 1, 1])
    s_vel_total = np.sqrt(Pe_total[:, 1, 1])
    
    s_curr_process = np.sqrt(Pe_process[:, 2, 2])
    s_curr_meas = np.sqrt(Pe_meas[:, 2, 2])
    s_curr_init = np.sqrt(Pe_init[:, 2, 2])
    s_curr_total = np.sqrt(Pe_total[:, 2, 2])
    
    # Standardavvik fra Kalman-filteret
    s_pos_kalman = np.sqrt(np.array([P_hat[k, 0, 0] for k in range(n)]))
    
    return (t, s_pos_process, s_pos_meas, s_pos_init, 
            s_vel_process, s_vel_meas, s_vel_init,
            s_curr_process, s_curr_meas, s_curr_init,
            s_pos_total, s_vel_total, s_curr_total, s_pos_kalman)

def plot_error_budget_results(results):
    """
    Plotter resultatene fra feilbudsjettet.
    
    Args:
        results: Resultatene fra error_budget-funksjonen
    """
    (t, s_pos_process, s_pos_meas, s_pos_init, 
     s_vel_process, s_vel_meas, s_vel_init,
     s_curr_process, s_curr_meas, s_curr_init,
     s_pos_total, s_vel_total, s_curr_total, s_pos_kalman) = results
    
    # Plott 1: Feilbudsjett for posisjon
    plt.figure(figsize=(10, 6))
    plt.plot(t, s_pos_process, 'r-', label='Prosesstøy')
    plt.plot(t, s_pos_meas, 'g-', label='Målingsstøy')
    plt.plot(t, s_pos_init, 'b-', label='Initialtilstand')
    plt.plot(t, s_pos_total, 'k--', label='Total (RMS)')
    plt.xlabel('Tid [s]')
    plt.ylabel('Standardavvik for posisjon [m]')
    plt.title('Feilbudsjett for posisjon')
    plt.legend()
    plt.grid(True)
    plt.savefig('task6_position_error_budget.png')
    
    # Plott 2: Feilbudsjett for hastighet
    plt.figure(figsize=(10, 6))
    plt.plot(t, s_vel_process, 'r-', label='Prosesstøy')
    plt.plot(t, s_vel_meas, 'g-', label='Målingsstøy')
    plt.plot(t, s_vel_init, 'b-', label='Initialtilstand')
    plt.plot(t, s_vel_total, 'k--', label='Total (RMS)')
    plt.xlabel('Tid [s]')
    plt.ylabel('Standardavvik for hastighet [m/s]')
    plt.title('Feilbudsjett for hastighet')
    plt.legend()
    plt.grid(True)
    plt.savefig('task6_velocity_error_budget.png')
    
    # Plott 3: Feilbudsjett for armaturstrøm
    plt.figure(figsize=(10, 6))
    plt.plot(t, s_curr_process, 'r-', label='Prosesstøy')
    plt.plot(t, s_curr_meas, 'g-', label='Målingsstøy')
    plt.plot(t, s_curr_init, 'b-', label='Initialtilstand')
    plt.plot(t, s_curr_total, 'k--', label='Total (RMS)')
    plt.xlabel('Tid [s]')
    plt.ylabel('Standardavvik for armaturstrøm [A]')
    plt.title('Feilbudsjett for armaturstrøm')
    plt.legend()
    plt.grid(True)
    plt.savefig('task6_current_error_budget.png')
    
    # Plott 4: Sammenligning av RMS-sum og Kalman-filter standardavvik
    plt.figure(figsize=(10, 6))
    plt.plot(t, s_pos_total, 'r-', label='Total (RMS-sum)')
    plt.plot(t, s_pos_kalman, 'b--', label='Kalman-filter')
    plt.xlabel('Tid [s]')
    plt.ylabel('Standardavvik for posisjon [m]')
    plt.title('RMS-sum vs. Kalman-filter standardavvik for posisjon')
    plt.legend()
    plt.grid(True)
    plt.savefig('task6_rms_vs_kalman.png')
    
    plt.show()

def main():
    # Systemparametre
    T2 = 5.0
    T3 = 1.0
    
    # Systemmatriser
    F = np.array([[0, 1, 0],
                 [0, -1/T2, 1/T2],
                 [0, 0, -1/T3]])
    
    L = np.array([[0], [0], [1/T3]])
    G = np.array([[0], [0], [1]])
    H = np.array([[1, 0, 0]])
    
    # Støyparametre
    Q_hat = 2 * 0.1**2
    R = 1
    
    # Initial kovarians
    P0 = np.diag([1, 0.1**2, 0.1**2])
    
    # Tidsparametre
    t0 = 0
    tf = 100
    dt = 0.01
    
    # Kjør feilbudsjett
    results = error_budget(F, G, H, Q_hat, R, P0, t0, tf, dt)
    
    # Plott resultater
    plot_error_budget_results(results)

if __name__ == "__main__":
    main()