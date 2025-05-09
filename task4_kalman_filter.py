import numpy as np
import matplotlib.pyplot as plt
from task1_simulation_model import t, n, P0, H, R
from task2_discretization import Phi, Lambda, Ga, S
from task3_simulation import x_s

# Generate measurements
u = 1.0
mi = 100  # Measurement interval (1 Hz = every 100 steps with dt=0.01)
nm = int(n / mi) + 1  # Number of measurements
k_meas = np.arange(0, n, mi)  # Times where measurements are available
z = np.zeros((1, nm))
for i in range(nm):
    k = k_meas[i]
    if k < n:  # Avoid index out of bounds
        z[:, i] = H @ x_s[:, k] + np.random.randn(1) * np.sqrt(R)

# Kalman filter implementation
def kalman_filter(z, Phi, Lambda, Ga, P0, H, R, mi, u, n):
    n_states = Phi.shape[0]
    
    # Initialize state vectors
    x_bar = np.zeros((n_states, n))  # A-priori estimates
    x_hat = np.zeros((n_states, nm))  # A-posteriori estimates at measurement times only
    
    # Initialize covariance matrices
    P_bar = np.zeros((n, n_states, n_states))  # A-priori error covariance
    P_hat = np.zeros((nm, n_states, n_states))  # A-posteriori error covariance
    
    # Initial conditions
    x_bar[:, 0] = np.zeros(n_states)
    P_bar[0] = P0
    
    # For storing measurement update times for plotting
    update_times = []
    
    # Process noise covariance
    S_k = Ga @ Ga.T
    
    # Kalman filter loop
    m_idx = 0  # Measurement index
    
    for k in range(n-1):
        # Measurement update (only at specific times)
        if k in k_meas:
            # Calculate Kalman gain
            K = P_bar[k] @ H.T @ np.linalg.inv(H @ P_bar[k] @ H.T + R)
            
            # Update state estimate with measurement
            x_hat[:, m_idx] = x_bar[:, k] + K @ (z[:, m_idx] - H @ x_bar[:, k])
            
            # Update error covariance
            P_hat[m_idx] = (np.eye(n_states) - K @ H) @ P_bar[k]
            
            # Store time of this update for plotting
            update_times.append(k)
            
            # Use this updated estimate for the time update
            x_next = Phi @ x_hat[:, m_idx] + Lambda * u
            P_next = Phi @ P_hat[m_idx] @ Phi.T + S_k
            
            m_idx += 1
        else:
            # No measurement, just propagate previous estimate
            x_next = Phi @ x_bar[:, k] + Lambda * u
            P_next = Phi @ P_bar[k] @ Phi.T + S_k
        
        # Store time update for next step
        x_bar[:, k+1] = x_next
        P_bar[k+1] = P_next
    
    # Compute standard deviations for plotting
    s_hat = np.zeros((nm,))
    s_bar = np.zeros((n,))
    
    for k in range(nm):
        s_hat[k] = np.sqrt(P_hat[k][1, 1])  # Velocity standard deviation (posteriori)
    
    for k in range(n):
        s_bar[k] = np.sqrt(P_bar[k][1, 1])  # Velocity standard deviation (priori)
    
    return x_bar, x_hat, s_bar, s_hat, update_times


# I task4_kalman_filter.py, legg til dette på slutten av filen:

if __name__ == "__main__":
    # Flytt all kode som genererer plott og kjører filteret her
    # For eksempel:
    u = 1.0
    mi = 100
    # Genererer målinger
    z = H @ x_s[:, k_meas] + np.random.randn(1, nm) * np.sqrt(R)

    # Run Kalman filter
    x_bar, x_hat, s_bar, s_hat, update_times = kalman_filter(z, Phi, Lambda, Ga, P0, H, R, mi, u, n)

    # Plotting
    plt.figure(figsize=(12, 10))

    # Plot 1: Velocity
    plt.subplot(3, 1, 1)
    plt.plot(t, x_s[1, :], label='$x_2$ (True Velocity)', alpha=0.5, color='blue')
    plt.plot(t, x_bar[1, :], label='$\\bar{x}_2$ (Predicted)', color='red', linestyle='--')

    # Plot the a-posteriori estimates at measurement times
    t_meas = t[update_times]
    plt.plot(t_meas, x_hat[1, :len(t_meas)], label='$\\hat{x}_2$ (Filtered)', color="green", markersize=4)

    plt.plot(t, u * np.ones_like(t), label='$u$', linestyle='--', alpha=0.5, color='black')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.title('Kalman Filter Velocity Estimates')

    # Plot 2: Filtered velocity errors
    plt.subplot(3, 1, 2)
    # Calculate filtered errors at measurement times
    filtered_errors = np.zeros(len(t_meas))
    for i, k in enumerate(update_times):
        if i < len(filtered_errors):  # Safety check
            filtered_errors[i] = x_s[1, k] - x_hat[1, i]

    plt.plot(t_meas, filtered_errors, label='$x_2 - \\hat{x}_2$', color='blue')
    plt.plot(t_meas, s_hat[:len(t_meas)], 'k--', label='$\\pm \\hat{s}_2$')
    plt.plot(t_meas, -s_hat[:len(t_meas)], 'k--')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Filtered Error')
    plt.title('Filtered Velocity Errors')

    # Plot 3: Predicted velocity errors
    plt.subplot(3, 1, 3)
    predicted_errors = x_s[1, :] - x_bar[1, :]
    plt.plot(t, predicted_errors, label='$x_2 - \\bar{x}_2$', color='blue')
    plt.plot(t, s_bar, 'k--', label='$\\pm \\bar{s}_2$')
    plt.plot(t, -s_bar, 'k--')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Predicted Error')
    plt.title('Predicted Velocity Errors')

    plt.tight_layout()
    plt.savefig('task4_kalman_filter.png')
    plt.show()