import numpy as np
import matplotlib.pyplot as plt

# ---------- 1D Chialvo-like map ----------
def chialvo_1d_step(x, alpha=3.9, beta=0.01):
    """One iteration of the 1D Chialvo-like map: x_{n+1} = alpha*x_n*(1-x_n) - beta."""
    return alpha * x * (1.0 - x) - beta

def chialvo_1d_trajectory(x0=0.5, alpha=3.9, beta=0.01, steps=5000, discard=500):
    """
    Generate trajectory for the 1D map.
    Returns the array AFTER discarding an initial transient.
    """
    x = np.empty(steps + discard, dtype=float)
    x[0] = x0
    for n in range(steps + discard - 1):
        x[n+1] = chialvo_1d_step(x[n], alpha=alpha, beta=beta)
    return x[discard:]

# ---------- Plot helpers ----------
def plot_time_series(x, N_show=800, title="Chaotic 1D Chialvo Map — Time Series"):
    plt.figure(figsize=(10, 4))
    N = min(N_show, len(x))
    plt.plot(np.arange(N), x[:N], lw=0.8)
    plt.xlabel("Iteration n")
    plt.ylabel("x_n")
    plt.title(title)
    plt.grid(True, alpha=0.3)

def plot_return_map(x, title="Chaotic 1D Chialvo Map — Return Plot"):
    plt.figure(figsize=(6, 6))
    plt.plot(x[:-1], x[1:], '.', markersize=1, alpha=0.5)
    plt.xlabel("x_n")
    plt.ylabel("x_{n+1}")
    plt.title(title)
    plt.grid(True, alpha=0.3)

def plot_bifurcation(alpha_min=3.5, alpha_max=4.0, n_alpha=800, beta=0.01,
                     steps=1500, discard=1000, keep_last=200, x0=0.5):
    """
    Bifurcation diagram sweeping alpha. For each alpha:
      - iterate the map, discard transient, then plot last 'keep_last' points.
    """
    alphas = np.linspace(alpha_min, alpha_max, n_alpha)
    xs_all = []
    alphas_all = []

    x = x0
    for a in alphas:
        # run trajectory for this alpha
        x = x0  # re-seed (or keep the previous x to speed up continuation)
        for _ in range(discard):
            x = chialvo_1d_step(x, alpha=a, beta=beta)
        # collect last few points
        xs = []
        for _ in range(keep_last):
            x = chialvo_1d_step(x, alpha=a, beta=beta)
            xs.append(x)
        xs_all.append(xs)
        alphas_all.append(np.full(keep_last, a))

    xs_all = np.concatenate(xs_all)
    alphas_all = np.concatenate(alphas_all)

    plt.figure(figsize=(10, 6))
    plt.plot(alphas_all, xs_all, ',',
             alpha=0.5)  # ',' is a super-tiny pixel marker
    plt.xlabel(r"$\alpha$")
    plt.ylabel("x")
    plt.title(f"Bifurcation Diagram (beta={beta})")
    plt.grid(True, alpha=0.2)

# ---------- Main ----------
if __name__ == "__main__":
    # Chaotic regime (tweak if you want): larger alpha (~3.6–4.0), small beta (~0–0.05)
    alpha = 3.9
    beta = 0.01
    x0 = 0.5

    # Generate trajectory
    traj = chialvo_1d_trajectory(x0=x0, alpha=alpha, beta=beta, steps=6000, discard=1000)

    # Visualizations
    plot_time_series(traj, N_show=800)
    plot_return_map(traj)
    plot_bifurcation(alpha_min=3.5, alpha_max=4.0, n_alpha=700, beta=beta,
                     steps=1800, discard=1200, keep_last=200, x0=x0)

    plt.tight_layout()
    plt.show()
