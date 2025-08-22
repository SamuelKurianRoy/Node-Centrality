import numpy as np
import matplotlib.pyplot as plt

def chialvo_map(x0, y0, params, n_iter=5000, discard=1000):
    """
    Simulate the Chialvo neuron model (discrete-time map).
    
    Args:
        x0, y0 (float): Initial conditions
        params (dict): Dictionary of model parameters {a, b, c, I}
        n_iter (int): Number of iterations
        discard (int): Number of transient steps to discard

    Returns:
        tuple: Arrays (x, y) of the simulated trajectory
    """
    a, b, c, I = params["a"], params["b"], params["c"], params["I"]

    x, y = np.zeros(n_iter), np.zeros(n_iter)
    x[0], y[0] = x0, y0

    for n in range(n_iter - 1):
        x[n+1] = x[n]**2 * np.exp(y[n] - x[n]) + I
        y[n+1] = a * y[n] - b * x[n] + c

    return x[discard:], y[discard:]

# --- Main simulation and plotting ---
if __name__ == "__main__":
    # Parameters (from Chialvo’s 1995 paper)
    params = {
        "a": 0.89,
        "b": 0.6,
        "c": 0.28,
        "I": 0.034
    }

    # Initial conditions
    x0, y0 = 0.1, 0.1

    print("Simulating the Chialvo neuron map...")

    x, y = chialvo_map(x0, y0, params, n_iter=20000, discard=2000)

    print("Simulation complete. Generating plots.")

    # 1. Time series of x
    plt.figure(figsize=(10, 4))
    plt.plot(x[:1000], lw=0.7, color="black")
    plt.title("Chialvo Neuron Time Series (x variable)", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("x", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 2. Phase space (x vs y)
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, ",", alpha=0.5, color="darkblue")
    plt.title("Chialvo Neuron Attractor (Phase Space)", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()
