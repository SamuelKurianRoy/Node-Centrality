import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Build Double-Star Symmetric Network
# ------------------------------------------------------------
def build_double_star_symmetric(n_leaves=6):
    """
    Build adjacency matrix for a symmetric double-star network:
    - Two hubs connected by a bridge
    - Each hub connects to n_leaves/2 leaves
    """
    num_nodes = n_leaves + 2 + 1  # two hubs + one bridge + leaves
    A = np.zeros((num_nodes, num_nodes))

    bridge = 0
    hub1 = 1
    hub2 = 2

    # Bridge connects to hubs
    A[bridge, hub1] = A[hub1, bridge] = 1
    A[bridge, hub2] = A[hub2, bridge] = 1

    # Leaves for hub1
    for i in range(3, 3 + n_leaves // 2):
        A[hub1, i] = A[i, hub1] = 1

    # Leaves for hub2
    for i in range(3 + n_leaves // 2, num_nodes):
        A[hub2, i] = A[i, hub2] = 1

    return A

# ------------------------------------------------------------
# Centrality Calculation
# ------------------------------------------------------------
def adjacent_edge_index(A):
    """Compute adjacent edge index (simple degree-based centrality)"""
    deg = A.sum(axis=1)
    total_deg = deg.sum()
    return deg / total_deg

def adjacent_edge_index_matrix(A):
    """Pairwise product centrality"""
    c = adjacent_edge_index(A)
    n = len(c)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = c[i] * c[j] * 2
    return M

# ------------------------------------------------------------
# 1D Logistic Map (Chaotic Dynamics)
# ------------------------------------------------------------
def chialvo_map_1d(x, r=3.9):
    """Chaotic logistic map"""
    return r * x * (1 - x)

def coupled_chialvo_step_1d(x, C, r=3.9):
    """
    One synchronous update step for 1D map with diffusive coupling
    """
    # Intrinsic update
    x_int = chialvo_map_1d(x, r=r)

    # Diffusive coupling
    inflow = C @ x
    outflow = (C.sum(axis=1)) * x
    x_coupled = x_int + inflow - outflow

    return x_coupled

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------
def simulate_network(A, control_node=None, gamma=0.2, n_steps=500, r=3.9):
    """
    Simulate coupled logistic maps with optional pinning control
    """
    num_nodes = A.shape[0]
    C = 0.04 * A  # base coupling strength

    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, size=num_nodes)

    X = np.zeros((n_steps, num_nodes))
    X[0, :] = x

    for t in range(1, n_steps):
        x = coupled_chialvo_step_1d(x, C, r=r)

        # Pinning control (drive selected node towards mean)
        if control_node is not None:
            x_mean = x.mean()
            x[control_node] += gamma * (x_mean - x[control_node])

        X[t, :] = x

    return X

# ------------------------------------------------------------
# Synchronization Error
# ------------------------------------------------------------
def synchronization_error(X):
    """
    Compute synchronization error = variance across nodes at each timestep
    """
    return np.var(X, axis=1)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Build network
    A = build_double_star_symmetric(n_leaves=10)

    # Centrality indices
    c = adjacent_edge_index(A)
    print("\n--- Adjacent Edge Index (Single Node Centrality) ---")
    labels = ["Bridge", "Hub1", "Hub2"] + [f"Leaf{i}" for i in range(1, A.shape[0]-2)]
    for i, val in enumerate(c):
        print(f"Node {i+1:2d} ({labels[i]:6}): Index = {val:.4f}")

    # Matrix
    M = adjacent_edge_index_matrix(A)
    print("\n--- Generated Adjacent Edge Index Matrix for Node Pairs (R^2) ---")
    print(np.round(M, 3))
    print("------------------------------------------------------------\n")

    # Simulations
    print("--- Running Synchronization Simulations (1D Logistic Map) ---")
    print("Using Base Coupling Strength: 0.04")
    print("Using Pinning Control Strength: 0.2\n")

    control_nodes = [0, 1, 3]  # bridge, hub1, leaf
    names = ["Control Bridge", "Control Hub", "Control Leaf"]

    results = {}
    for idx, cn in zip(names, control_nodes):
        print(f"Simulating: {idx} (Node {cn+1})...")
        X = simulate_network(A, control_node=cn, gamma=0.2, n_steps=500)
        err = synchronization_error(X)
        results[idx] = err

    print("\nSimulations complete. Plotting results...")

    # Plot
    plt.figure(figsize=(10, 6))
    for k, v in results.items():
        plt.semilogy(v, label=k)
    plt.xlabel("Time step")
    plt.ylabel("Synchronization Error (log scale)")
    plt.title("Synchronization under Pinning Control (1D Logistic Map)")
    plt.legend()
    plt.grid(True)
    plt.show()
