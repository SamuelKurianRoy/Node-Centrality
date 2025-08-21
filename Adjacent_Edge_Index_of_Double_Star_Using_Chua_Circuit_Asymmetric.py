import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. Define the Double-Star Network ---
def create_double_star_network():
    """
    Creates the 13-node double-star network from Figure 3(a) in the paper.
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
    """
    Calculates the 'adjacent edge index' (R_i) for each node in the graph.
    This is a required preliminary step for the node-pair calculation.
    """
    # Get the Laplacian matrix. Node order is fixed for consistency.
    nodelist = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist).toarray()

    # Eigen-decomposition to find the Fiedler vector
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1] # Vector for the 2nd smallest eigenvalue

    # Calculate the index R_i = sum(|x_i - x_j|) for all neighbors j of i
    indices = {}
    for i in nodelist:
        r_i = 0
        xi = fiedler_vector[i-1] # 0-based index for vector
        for neighbor in G.neighbors(i):
            xj = fiedler_vector[neighbor-1]
            r_i += np.abs(xi - xj)
        indices[i] = r_i

    return indices, fiedler_vector

# --- 3. Calculate the Adjacent Edge Index Matrix (for node pairs) ---
def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector):
    """
    Generates the R^2 matrix for node pairs using the user-specified equation.

    The formula is: R_jk^2 = (Sum over neighbors p of j of |x_j - x_p|)
                             + (Sum over neighbors q of k of |x_k - x_q|)
                             - |E_j intersect E_k| * |x_j - x_k|

    This simplifies to: R_jk^2 = R_j + R_k - (correction_term)
    """
    num_nodes = G.number_of_nodes()
    nodelist = sorted(G.nodes())

    R2_matrix = np.zeros((num_nodes, num_nodes))

    for j_idx, node_j in enumerate(nodelist):
        for k_idx, node_k in enumerate(nodelist):
            if j_idx >= k_idx:
                continue # Only compute the upper triangle of the matrix

            # The first part of the formula is the pre-calculated index R_j.
            R_j = single_node_indices[node_j]

            # The second part of the formula is the pre-calculated index R_k.
            R_k = single_node_indices[node_k]

            # The third part is the correction term for adjacent nodes.
            # |E_j intersect E_k| is 1 if nodes j and k share an edge, and 0 otherwise.
            intersection_count = 1 if G.has_edge(node_j, node_k) else 0

            x_j = fiedler_vector[j_idx]
            x_k = fiedler_vector[k_idx]

            # This is the |x_j - x_k| part of the correction term.
            fiedler_difference = np.abs(x_j - x_k)

            correction_term = intersection_count * fiedler_difference

            # Combine the terms to get the final value for the pair (j,k)
            R_jk = R_j + R_k - correction_term

            R2_matrix[j_idx, k_idx] = R_jk

    return R2_matrix

# --- 4. Define Chua's Circuit and Network Dynamics ---
def chua_dynamics(t, state):
    """Defines the ODEs for a single Chua's circuit."""
    p, q, r = state
    gamma1, gamma2, gamma3, gamma4 = 10.0, 18.0, -4.0/3.0, -3.0/4.0
    g_p = gamma4 * p + 0.5 * (gamma3 - gamma4) * (abs(p + 1) - abs(p - 1))
    return [-gamma1 * (p - q + g_p), p - q + r, -gamma2 * q]

def coupled_network_dynamics(t, network_state, G, L, controlled_nodes, c_base, c_pin):
    """Defines the dynamics for the entire network of coupled Chua's circuits."""
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
    # --- Part 1 & 2: Build network and calculate single node indices ---
    G = create_double_star_network()
    single_node_indices, fiedler_vector = calculate_single_node_index(G)
    pos = nx.spring_layout(G, seed=42)  # layout for better positioning
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

    # --- Part 3: Generate and display the Node Pair Matrix ---
    R2_matrix = calculate_node_pair_matrix(G, single_node_indices, fiedler_vector)
    print("\n--- Generated Adjacent Edge Index Matrix for Node Pairs (R^2) ---")
    np.set_printoptions(precision=3, suppress=True)
    print(R2_matrix)
    print("-" * 60)

    # --- Part 4 & 5: Simulate network synchronization ---
    print("\n--- Running Synchronization Simulations ---")

    # Define coupling strengths similar to, but not the same as, the paper
    base_coupling_strength = 2.5
    pinning_coupling_strength = 5 * base_coupling_strength # Enhanced strength for control
    print(f"Using Base Coupling Strength: {base_coupling_strength}")
    print(f"Using Pinning Control Strength: {pinning_coupling_strength}\n")

    num_nodes = G.number_of_nodes()
    t_span = [0, 12] # Extended time slightly for clearer convergence
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

    # --- Plotting the results ---
    print("\nSimulations complete. Plotting results...")
    plt.figure(figsize=(12, 7))

    for name, states in results.items():
        states = states.reshape(num_nodes, 3, -1)
        avg_state = np.mean(states, axis=0)
        error = np.linalg.norm(states - avg_state, axis=1) # Calculate norm for each node
        total_error = np.mean(error, axis=0) # Average the norms
        plt.plot(t_eval, total_error, label=name)

    plt.title("Synchronization of Double-Star Network with Controlled Nodes", fontsize=16)
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Average Synchronization Error", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.ylim(bottom=1e-5)
    plt.show()