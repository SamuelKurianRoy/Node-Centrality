import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Double-Star Network ---
def create_double_star_network():
    """
    Creates the 13-node double-star network from Figure 3(a).
    Node 1 is the bridge, nodes 2 and 8 are hubs.
    """
    G = nx.Graph()
    nodes = range(1, 14)
    G.add_nodes_from(nodes)

    # First star (hub: 2) connected to bridge (1)
    G.add_edge(1, 2)
    for i in range(3, 8):
        G.add_edge(2, i)

    # Second star (hub: 8) connected to bridge (1)
    G.add_edge(1, 8)
    for i in range(9, 14):
        G.add_edge(8, i)

    return G

# --- 2. Calculate the Adjacent Edge Index (for single nodes) ---
def calculate_single_node_index(G):
    """Calculates the 'adjacent edge index' (R_i) for each node."""
    nodelist = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist).toarray()

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]  # 2nd smallest eigenvalue vector

    indices = {}
    for i in nodelist:
        r_i = 0
        xi = fiedler_vector[i - 1]
        for neighbor in G.neighbors(i):
            xj = fiedler_vector[neighbor - 1]
            r_i += np.abs(xi - xj)
        indices[i] = r_i

    return indices, fiedler_vector

# --- 3. Calculate the Adjacent Edge Index Matrix (for node pairs) ---
def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector):
    """Generates the R^2 matrix for node pairs."""
    num_nodes = G.number_of_nodes()
    nodelist = sorted(G.nodes())
    R2_matrix = np.zeros((num_nodes, num_nodes))

    for j_idx, node_j in enumerate(nodelist):
        for k_idx, node_k in enumerate(nodelist):
            if j_idx >= k_idx:
                continue
            R_j = single_node_indices[node_j]
            R_k = single_node_indices[node_k]
            intersection_count = 1 if G.has_edge(node_j, node_k) else 0
            x_j = fiedler_vector[j_idx]
            x_k = fiedler_vector[k_idx]
            fiedler_difference = np.abs(x_j - x_k)
            correction_term = intersection_count * fiedler_difference
            R_jk = R_j + R_k - correction_term
            R2_matrix[j_idx, k_idx] = R_jk
    return R2_matrix

# --- 4. Chialvo neuron (single-node map) ---
def chialvo_map(x, y, a=0.89, b=0.6, c=0.28, I=0.034):
    """
    Vectorized Chialvo neuron map for arrays x, y.
    Uses clipping to avoid exponential overflow.
    """
    exp_term = np.exp(np.clip(y - x, -10, 10))  # prevent overflow
    x_next = x**2 * exp_term + I
    y_next = a * y - b * x + c
    return x_next, y_next


# --- 5. Coupled network dynamics (discrete-time map with diffusive x-coupling) ---
def build_coupling_matrix(G, controlled_nodes, c_base, c_pin):
    """
    Returns a symmetric matrix C where C[i,j] is the coupling strength on edge (i,j),
    else 0. Edges incident to any controlled node use c_pin, others use c_base.
    """
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    C = np.zeros_like(A)

    # base coupling on edges
    C[A > 0] = c_base

    # pinning: edges incident to any controlled node get stronger coupling
    controlled_set = set(controlled_nodes)
    for u, v in G.edges():
        if (u in controlled_set) or (v in controlled_set):
            i, j = idx[u], idx[v]
            C[i, j] = c_pin
            C[j, i] = c_pin
    return C

def coupled_chialvo_step(x, y, C, a=0.89, b=0.6, c=0.28, I=0.034):
    """
    One synchronous update step for the entire network.
    Diffusive coupling acts on x only: sum_j C_ij (x_j - x_i).
    """
    # Intrinsic update
    x_int, y_int = chialvo_map(x, y, a=a, b=b, c=c, I=I)

    # Diffusive coupling term for x
    # term_i = sum_j C_ij * x_j - x_i * sum_j C_ij
    inflow = C @ x
    outflow = (C.sum(axis=1)) * x
    x_coupled = x_int + inflow - outflow

    # y is usually uncoupled in this map (fast variable x is coupled)
    y_coupled = y_int

    return x_coupled, y_coupled

# --- Main execution block ---
if __name__ == '__main__':
    # Part 1 & 2: Build network and calculate indices
    G = create_double_star_network()
    single_node_indices, fiedler_vector = calculate_single_node_index(G)

    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1000,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray"
    )
    plt.title("Double-Star Network", fontsize=14)
    plt.show()

    print("--- Adjacent Edge Index (Single Node Centrality) ---")
    sorted_indices = sorted(single_node_indices.items(), key=lambda item: item[1], reverse=True)
    for node, index in sorted_indices:
        role = "Bridge" if node == 1 else "Hub" if node in [2, 8] else "Leaf"
        print(f"Node {node:2d} ({role:^6s}): Index = {index:.4f}")

    # Part 3: Generate and display the Node Pair Matrix
    R2_matrix = calculate_node_pair_matrix(G, single_node_indices, fiedler_vector)
    print("\n--- Generated Adjacent Edge Index Matrix for Node Pairs (R^2) ---")
    np.set_printoptions(precision=3, suppress=True)
    print(R2_matrix)
    print("-" * 60)

    # Part 4 & 5: Simulate network synchronization with Chialvo neurons (discrete map)
    print("\n--- Running Synchronization Simulations (Chialvo Map) ---")
    base_coupling_strength = 0.04   # smaller typical values for maps
    pinning_coupling_strength = 5 * base_coupling_strength
    print(f"Using Base Coupling Strength: {base_coupling_strength}")
    print(f"Using Pinning Control Strength: {pinning_coupling_strength}\n")

    num_nodes = G.number_of_nodes()
    n_steps = 3000
    discard = 200  # transient to discard for plotting error
    nodes_sorted = sorted(G.nodes())

    # Chialvo parameters (can tweak for different regimes)
    a, b, c, I = 0.89, 0.6, 0.28, 0.034

    # Random initial conditions
    rng = np.random.default_rng(42)
    x0 = rng.uniform(-0.5, 0.5, size=num_nodes)
    y0 = rng.uniform(-0.5, 0.5, size=num_nodes)

    control_scenarios = {
        "Control Bridge (Node 1)": [1],
        "Control Hub (Node 2)": [2],
        "Control Leaf (Node 3)": [3],
    }

    # Run all scenarios
    results = {}
    for name, nodes_to_control in control_scenarios.items():
        print(f"Simulating: {name}...")
        # Build coupling matrix for this scenario
        C = build_coupling_matrix(G, nodes_to_control, base_coupling_strength, pinning_coupling_strength)

        # Allocate arrays to store trajectories
        X = np.zeros((n_steps, num_nodes))
        Y = np.zeros((n_steps, num_nodes))
        X[0, :] = x0
        Y[0, :] = y0

        x, y = x0.copy(), y0.copy()
        for t in range(1, n_steps):
            x, y = coupled_chialvo_step(x, y, C, a=a, b=b, c=c, I=I)
            X[t, :] = x
            Y[t, :] = y

        results[name] = (X, Y)

    # --- Plotting synchronization error per node over iterations ---
    print("\nSimulations complete. Plotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    iters = np.arange(n_steps - discard)

    for ax, (name, (X, Y)) in zip(axes, results.items()):
        # discard transient
        Xs = X[discard:, :]   # shape: [T, N]
        Ys = Y[discard:, :]

        # average state across nodes at each iteration
        x_mean = Xs.mean(axis=1, keepdims=True)  # [T, 1]
        y_mean = Ys.mean(axis=1, keepdims=True)  # [T, 1]

        # synchronization error per node: Euclidean norm of deviation in (x, y)
        # errors shape: [T, N]
        dx = Xs - x_mean
        dy = Ys - y_mean
        errors = np.sqrt(dx**2 + dy**2)

        # plot each node's error trajectory
        for node_idx in range(num_nodes):
            ax.plot(iters, errors[:, node_idx], alpha=0.6, linewidth=1, label=f"Node {node_idx+1}")

        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Iteration (n)", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("Synchronization Error per Node", fontsize=12)
    axes[-1].legend(fontsize=8, loc="upper right", ncol=2)
    plt.suptitle("Synchronization of Double-Star Network with Chialvo Neurons (All Nodes)", fontsize=16)
    plt.ylim(bottom=1e-6)
    plt.show()
