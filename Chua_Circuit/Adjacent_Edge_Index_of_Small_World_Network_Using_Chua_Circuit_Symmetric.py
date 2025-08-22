import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. Define a Small-World Network (Watts–Strogatz) ---
def create_small_world_network(n=20, k=4, p=0.2, seed=42):
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return G

# --- 1b. Optional: Compute small-world metrics ---
def compute_small_world_metrics(G):
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)

    n = G.number_of_nodes()
    k = int(np.mean([deg for _, deg in G.degree()]))

    # Generate a comparable random graph
    G_rand = nx.gnm_random_graph(n, int(n * k / 2), seed=42)

    # Ensure connectivity: take the largest connected component
    if not nx.is_connected(G_rand):
        largest_cc = max(nx.connected_components(G_rand), key=len)
        G_rand = G_rand.subgraph(largest_cc).copy()

    C_rand = nx.average_clustering(G_rand)
    L_rand = nx.average_shortest_path_length(G_rand)

    S = (C / C_rand) / (L / L_rand)
    return C, L, C_rand, L_rand, S

# --- 2. Calculate the Adjacent Edge Index (for single nodes) ---
def calculate_single_node_index(G):
    nodelist = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]  # Fiedler vector

    indices = {}
    for i in nodelist:
        r_i = 0
        xi = fiedler_vector[i - 1]  # adjust to 0-based
        for neighbor in G.neighbors(i):
            xj = fiedler_vector[neighbor - 1]
            r_i += np.abs(xi - xj)
        indices[i] = r_i

    return indices, fiedler_vector

# --- 3. Calculate the Adjacent Edge Index Matrix (for node pairs) ---
def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector):
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

# --- 4. Define Chua's Circuit and Network Dynamics ---
def chua_dynamics(t, state):
    p, q, r = state
    gamma1, gamma2, gamma3, gamma4 = 10.0, 18.0, -4.0/3.0, -3.0/4.0
    g_p = gamma4 * p + 0.5 * (gamma3 - gamma4) * (abs(p + 1) - abs(p - 1))
    return [-gamma1 * (p - q + g_p), p - q + r, -gamma2 * q]

def coupled_network_dynamics(t, network_state, G, L, controlled_nodes, c_base, c_pin):
    num_nodes = G.number_of_nodes()
    network_state = network_state.reshape((num_nodes, 3))
    d_dt = np.zeros_like(network_state)
    coupling_term = np.zeros_like(network_state)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if L[i, j] != 0:
                is_controlled = ((i + 1) in controlled_nodes or (j + 1) in controlled_nodes)
                c = c_pin if is_controlled else c_base
                coupling_term[i] -= c * L[i, j] * network_state[j]

    for i in range(num_nodes):
        d_dt[i] = chua_dynamics(t, network_state[i]) + coupling_term[i]

    return d_dt.flatten()

# --- Main execution block ---
if __name__ == '__main__':
    # --- Part 1: Build small-world network ---
    G = create_small_world_network(n=20, k=4, p=0.2)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=600,
        node_color="skyblue",
        font_size=9,
        font_weight="bold",
        edge_color="gray"
    )
    plt.title("Watts–Strogatz Small-World Network", fontsize=14)
    plt.show()

    # Compute small-world metrics
    C, L, C_rand, L_rand, S = compute_small_world_metrics(G)
    print("--- Small-World Metrics ---")
    print(f"Clustering Coefficient C = {C:.3f} (Random: {C_rand:.3f})")
    print(f"Average Path Length L    = {L:.3f} (Random: {L_rand:.3f})")
    print(f"Small-World Index S      = {S:.3f}")
    print("-" * 50)

    # --- Part 2: Single node indices ---
    single_node_indices, fiedler_vector = calculate_single_node_index(G)
    print("--- Adjacent Edge Index (Single Node Centrality) ---")
    sorted_indices = sorted(single_node_indices.items(), key=lambda item: item[1], reverse=True)
    for node, index in sorted_indices:
        print(f"Node {node:2d}: Index = {index:.4f}")

    # --- Part 3: Node Pair Matrix ---
    R2_matrix = calculate_node_pair_matrix(G, single_node_indices, fiedler_vector)
    print("\n--- Generated Adjacent Edge Index Matrix for Node Pairs (R^2) ---")
    np.set_printoptions(precision=3, suppress=True)
    print(R2_matrix)
    print("-" * 60)

    # --- Part 4 & 5: Synchronization Simulations ---
    print("\n--- Running Synchronization Simulations ---")
    base_coupling_strength = 2.5
    pinning_coupling_strength = 5 * base_coupling_strength
    print(f"Using Base Coupling Strength: {base_coupling_strength}")
    print(f"Using Pinning Control Strength: {pinning_coupling_strength}\n")

    num_nodes = G.number_of_nodes()
    t_span = [0, 12]
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    np.random.seed(42)
    initial_conditions = (np.random.rand(num_nodes * 3) - 0.5) * 10
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray()

    control_scenarios = {
        "Control Node 1": [1],
        "Control Node 2": [2],
        "Control Node 3": [3],
    }

    results = {}
    for name, nodes_to_control in control_scenarios.items():
        print(f"Simulating: {name}...")
        sol = solve_ivp(
            fun=coupled_network_dynamics, t_span=t_span, y0=initial_conditions,
            t_eval=t_eval, args=(G, L, nodes_to_control, base_coupling_strength, pinning_coupling_strength)
        )
        results[name] = sol.y

    # --- Plot results (per-node errors instead of average only) ---
    print("\nSimulations complete. Plotting results...")
    plt.figure(figsize=(12, 7))

    for name, states in results.items():
        states = states.reshape(num_nodes, 3, -1)

        # Calculate synchronization error for each node
        avg_state = np.mean(states, axis=0)
        errors = np.linalg.norm(states - avg_state, axis=1)  # shape: (num_nodes, time_points)

        # Plot each node’s trajectory
        for node_idx in range(num_nodes):
            plt.plot(
                t_eval, errors[node_idx],
                alpha=0.6, linewidth=1,
                label=f"{name} - Node {node_idx+1}" if node_idx == 0 else None
            )

    plt.title("Synchronization in Small-World Network (All Nodes)", fontsize=16)
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Synchronization Error per Node", fontsize=12)
    plt.legend(fontsize=8, loc="upper right", ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.ylim(bottom=1e-5)
    plt.show()
