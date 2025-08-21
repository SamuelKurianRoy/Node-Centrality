import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def chua_dynamics(t, state):
    """
    Defines the system of ordinary differential equations for Chua's circuit.
    These are the dynamics for each individual node in the network.

    Args:
        t (float): The current time (not used in this autonomous system, but required by the solver).
        state (list or array): A list of the three state variables [p, q, r].

    Returns:
        list: The derivatives [dp/dt, dq/dt, dr/dt].
    """
    p, q, r = state

    # Parameters from the paper's Figure 2 caption (Section 4.1)
    gamma1, gamma2 = 10.0, 18.0
    gamma3, gamma4 = -4.0 / 3.0, -3.0 / 4.0

    # Nonlinear component g(p) as defined in the paper
    g_p = gamma4 * p + 0.5 * (gamma3 - gamma4) * (abs(p + 1) - abs(p - 1))

    # The system of ODEs from Equation (8) in the paper
    dp_dt = -gamma1 * (p - q + g_p)
    dq_dt = p - q + r
    dr_dt = -gamma2 * q

    return [dp_dt, dq_dt, dr_dt]

# --- Main simulation and plotting block ---
if __name__ == '__main__':
    # Initial conditions from the paper's Figure 2 caption
    # A small non-zero initial state is needed to start the chaotic dynamics
    initial_state = [0.1, 0.1, 0.1]

    # Time span for the simulation
    # A long duration is needed to allow the attractor to fully form
    t_span = [0, 100]

    # Generate points in time where the solution will be saved
    # More points will result in a smoother plot
    t_eval = np.linspace(t_span[0], t_span[1], 10000)

    print("Simulating the Chua circuit dynamics...")

    # Use SciPy's ODE solver to integrate the system of equations
    solution = solve_ivp(
        fun=chua_dynamics,
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        dense_output=True, # Allows for smooth plotting
        rtol=1e-6,         # Relative tolerance for accuracy
        atol=1e-9          # Absolute tolerance for accuracy
    )

    print("Simulation complete. Generating plots.")

    # Extract the state variables from the solution object
    p, q, r = solution.y

    # --- Plotting ---

    # 1. Create the 2D plot (p vs q) to replicate Figure 2 from the paper
    plt.figure(figsize=(8, 6))
    plt.plot(p, q, lw=0.5, color='black')
    plt.title("Chua's Circuit Attractor (2D Projection)", fontsize=16)
    plt.xlabel("p", fontsize=12)
    plt.ylabel("q", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.figtext(0.5, 0.01, "This plot replicates the chaotic attractor shown in Figure 2 of the paper.", ha="center", fontsize=9)

    # 2. Create the 3D plot for a better visualization of the attractor
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p, q, r, lw=0.5, color='mediumblue')
    ax.set_title("Chua's Circuit 'Double-Scroll' Attractor (3D)", fontsize=16)
    ax.set_xlabel("p", fontsize=12)
    ax.set_ylabel("q", fontsize=12)
    ax.set_zlabel("r", fontsize=12)

    # Show both plots
    plt.show()