import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. Define the Double-Star Network ---
def create_double_star_network():
    """
    Creates the 15-node double-star network.
    Node 1 is the bridge, nodes 2 and 8 are hubs.
    """
    G = nx.Graph()
    nodes = range(1, 16)  # Now 15 nodes
    G.add_nodes_from(nodes)

    # First star (hub: 2) connected to bridge (1)
    G.add_edge(1, 2)
    for i in range(3, 8):
        G.add_edge(2, i)

    # Second star (hub: 8) connected to bridge (1)
    G.add_edge(1, 8)
    for i in range(9, 16):
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

# --- 4. Define Chua's Circuit and Network Dynamics ---
def chua_dynamics(t, state):
    """ODEs for a single Chua's circuit."""
    p, q, r = state
    gamma1, gamma2, gamma3, gamma4 = 10.0, 18.0, -4.0/3.0, -3.0/4.0
    g_p = gamma4 * p + 0.5 * (gamma3 - gamma4) * (abs(p + 1) - abs(p - 1))
    return [-gamma1 * (p - q + g_p), p - q + r, -gamma2 * q]

def coupled_network_dynamics(t, network_state, G, L, controlled_nodes, c_base, c_pin):
    """Dynamics for the entire network of coupled Chua's circuits."""
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

    # Part 3: Node Pair Matrix
    R2_matrix = calculate_node_pair_matrix(G, single_node_indices, fiedler_vector)
    print("\n--- Generated Adjacent Edge Index Matrix for Node Pairs (R^2) ---")
    np.set_printoptions(precision=3, suppress=True)
    print(R2_matrix)
    print("-" * 60)

    # Part 4 & 5: Synchronization simulations
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
        "Control Bridge (Node 1)": [1],
        "Control Hub (Node 2)": [2],
        "Control Leaf (Node 3)": [3],
    }

    results = {}
    for name, nodes_to_control in control_scenarios.items():
        print(f"Simulating: {name}...")
        sol = solve_ivp(
            fun=coupled_network_dynamics, t_span=t_span, y0=initial_conditions,
            t_eval=t_eval, args=(G, L, nodes_to_control, base_coupling_strength, pinning_coupling_strength)
        )
        results[name] = sol.y

    # --- Plotting all nodes per scenario ---
    print("\nSimulations complete. Plotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (name, states) in zip(axes, results.items()):
        states = states.reshape(num_nodes, 3, -1)
        avg_state = np.mean(states, axis=0)
        errors = np.linalg.norm(states - avg_state, axis=1)

        for node_idx in range(num_nodes):
            ax.plot(t_eval, errors[node_idx], alpha=0.6, linewidth=1, label=f"Node {node_idx+1}")

        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Time (t)", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("Synchronization Error per Node", fontsize=12)
    axes[-1].legend(fontsize=8, loc="upper right", ncol=2)
    plt.suptitle("Synchronization of Double-Star Network (All Nodes)", fontsize=16)
    plt.ylim(bottom=1e-5)
    plt.show()
