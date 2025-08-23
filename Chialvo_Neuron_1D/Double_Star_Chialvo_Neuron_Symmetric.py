import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Define Chialvo Neuron (1D Map) ---
def chialvo_update(x, alpha=3.9, beta=0.01):
    """One iteration of Chialvo neuron dynamics (bounded)."""
    x_new = alpha * x * (1 - x) - beta
    # keep values bounded (avoid overflow)
    if np.isnan(x_new) or np.isinf(x_new):
        x_new = 0.5
    return np.clip(x_new, -2, 2)

# --- 2. Create Double-Star Network ---
def create_double_star_network():
    G = nx.Graph()
    nodes = range(1, 14)
    G.add_nodes_from(nodes)
    G.add_edge(1, 2)
    for i in range(3, 8):
        G.add_edge(2, i)
    G.add_edge(1, 8)
    for i in range(9, 14):
        G.add_edge(8, i)
    return G

# --- 3. Simulate Coupled Chialvo Neurons ---
def simulate_network(G, steps=2000, alpha=3.9, beta=0.01, coupling=0.05):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    x = np.random.rand(n)  # initial states
    states = []

    for t in range(steps):
        x_new = np.zeros_like(x)
        for i in range(n):
            xi = chialvo_update(x[i], alpha, beta)
            coupling_term = coupling * np.sum(A[i] * (x - x[i]))
            x_new[i] = xi + coupling_term
        x = np.clip(x_new, -2, 2)
        states.append(x.copy())

    return np.array(states)

# --- 4. Adjacent Edge Index from Chialvo states ---
def calculate_indices(G, avg_states):
    indices = {}
    for i in G.nodes():
        r_i = 0
        for j in G.neighbors(i):
            r_i += abs(avg_states[i-1] - avg_states[j-1])
        indices[i] = r_i
    return indices

def calculate_pair_matrix(G, single_node_indices, avg_states):
    n = G.number_of_nodes()
    nodelist = sorted(G.nodes())
    R2_matrix = np.zeros((n, n))
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i >= j:
                continue
            R_u = single_node_indices[u]
            R_v = single_node_indices[v]
            correction = abs(avg_states[i] - avg_states[j]) if G.has_edge(u, v) else 0
            R2_matrix[i, j] = R_u + R_v - correction
    return R2_matrix

# --- MAIN ---
if __name__ == '__main__':
    G = create_double_star_network()
    pos = nx.spring_layout(G, seed=42)

    # Draw initial network
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", edge_color="gray")
    plt.title("Double-Star Network (Initial)", fontsize=14)
    plt.show()

    # Simulate Chialvo neurons
    states = simulate_network(G, steps=3000, coupling=0.05)
    avg_states = np.mean(states[1000:], axis=0)  # discard transient

    # Compute indices
    single_node_indices = calculate_indices(G, avg_states)
    print("Adjacent Edge Index (R_i) for each node:")
    for node, idx in single_node_indices.items():
        print(f"Node {node}: {idx:.4f}")

    # Compute R² matrix
    R2_matrix = calculate_pair_matrix(G, single_node_indices, avg_states)
    nodelist = sorted(G.nodes())
    R2_df = pd.DataFrame(R2_matrix, index=nodelist, columns=nodelist)

    print("\nR² Matrix (for node pairs):")
    for i, row in enumerate(R2_matrix):
        row_str = ", ".join([f"{val:.4f}" if j >= i else "-----" for j, val in enumerate(row)])
        print(f"{nodelist[i]:2d} [{row_str}]")

    # Find most important node & pair
    most_important_node = max(single_node_indices, key=single_node_indices.get)
    max_val = np.max(R2_matrix)
    max_pos = np.unravel_index(np.argmax(R2_matrix), R2_matrix.shape)
    max_row, max_col = nodelist[max_pos[0]], nodelist[max_pos[1]]

    print("\nMost Important Node:", most_important_node)
    print(f"Most Important Pair: ({max_row}, {max_col}) with R² = {max_val:.4f}")

    # Highlight in network
    node_colors = ["skyblue"] * len(G.nodes())
    node_colors[nodelist.index(most_important_node)] = "red"
    for node in (max_row, max_col):
        if node != most_important_node:
            node_colors[nodelist.index(node)] = "green"

    edge_colors = ["green" if set([u, v]) == set([max_row, max_col]) else "gray" for u, v in G.edges()]

    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, edge_color=edge_colors, width=2)
    plt.title("Double-Star Network (Highlighted Node & Pair)", fontsize=14)
    plt.show()
