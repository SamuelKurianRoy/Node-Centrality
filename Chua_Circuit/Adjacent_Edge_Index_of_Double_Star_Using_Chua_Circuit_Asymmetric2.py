import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Define the Double-Star Network ---
def create_double_star_network():
    """
    Creates the 13-node double-star network.
    Node 1 is the bridge, nodes 2 and 8 are hubs.
    """
    G = nx.Graph()
    nodes = range(1, 17)
    G.add_nodes_from(nodes)

    # First star (hub: 2) connected to bridge (1)
    G.add_edge(1, 2)
    for i in range(3, 12):
        G.add_edge(2, i)

    # Second star (hub: 8) connected to bridge (1)
    G.add_edge(1, 8)
    for i in range(12, 17):
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
    """Generates the R² matrix for node pairs."""
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

# --- Main execution block ---
if __name__ == '__main__':
    # Step 1: Build network
    G = create_double_star_network()
    pos = nx.spring_layout(G, seed=42)

    # Draw plain blue network first
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1000,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray"
    )
    plt.title("Double-Star Network (Initial)", fontsize=14)
    plt.show()

    # Step 2: Calculate node indices
    single_node_indices, fiedler_vector = calculate_single_node_index(G)

    print("Adjacent Edge Index (R_i) for each node:")
    for node, index in single_node_indices.items():
        print(f"Node {node}: {index:.4f}")

    # Find most important node
    most_important_node = max(single_node_indices, key=single_node_indices.get)

    # Step 3: Generate and display the Node Pair Matrix
    R2_matrix = calculate_node_pair_matrix(G, single_node_indices, fiedler_vector)
    nodelist = sorted(G.nodes())
    R2_df = pd.DataFrame(R2_matrix, index=nodelist, columns=nodelist)

    # Find maximum R² value and position
    max_val = np.max(R2_matrix)
    max_pos = np.unravel_index(np.argmax(R2_matrix), R2_matrix.shape)
    max_row, max_col = nodelist[max_pos[0]], nodelist[max_pos[1]]

    print("\nR² Matrix (for node pairs):")
    for i, row in enumerate(R2_df.values):
        row_str = []
        for j, val in enumerate(row):
            if (i, j) == max_pos:
                # Highlight maximum with a star and green color
                row_str.append(f"\033[1;32m{val:.4f}*\033[0m")
            else:
                row_str.append(f"{val:.4f}")
        print(f"{nodelist[i]:2d} [{', '.join(row_str)}]")

    most_important_pair = (max_row, max_col)

    print(f"\nMost Important Node: {most_important_node}")
    print(f"Most Important Pair: {most_important_pair} with R² = {max_val:.4f}")

    # Step 4: Draw network with highlighted nodes & pair
    node_colors = ["skyblue"] * len(G.nodes())
    node_colors[nodelist.index(most_important_node)] = "red"

    for node in most_important_pair:
        if node != most_important_node:  # don't override red node
            node_colors[nodelist.index(node)] = "green"

    edge_colors = []
    for u, v in G.edges():
        if set([u, v]) == set(most_important_pair):
            edge_colors.append("green")
        else:
            edge_colors.append("gray")

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1000,
        node_color=node_colors,
        font_size=10,
        font_weight="bold",
        edge_color=edge_colors,
        width=2
    )
    plt.title("Double-Star Network (Highlighted Node & Pair)", fontsize=14)
    plt.show()
