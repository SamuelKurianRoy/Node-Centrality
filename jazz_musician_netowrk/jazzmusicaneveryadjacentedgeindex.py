
import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd

# Define the path to your network file
file_path = 'Jazz-Musicians-Network.wl'

if os.path.exists(file_path):
    # Read the content of the file
    with open(file_path, 'r') as f:
        file_content = f.read()

    # Extract edges using regular expressions
    edges = re.findall(r'UndirectedEdge\[(\d+),\s*(\d+)\]', file_content)

    # Create a graph from the extracted edges
    G_edge = nx.Graph()
    G_edge.add_edges_from([(int(u), int(v)) for u, v in edges])

    print("Network Information:")
    print(f"Number of nodes: {G_edge.number_of_nodes()}")
    print(f"Number of edges: {G_edge.number_of_edges()}")
    print("-" * 20)

    # --- Visualize the network ---
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(G_edge)
    nx.draw(G_edge, pos, with_labels=True, node_color='skyblue', node_size=200, 
            edge_color='gray', font_size=8, font_weight='bold')
    plt.title("Jazz Musicians Network Visualization", size=20)
    plt.show()

else:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure your .wl file is in the same directory as this script.")

# ---------- Centrality calculations ----------
def calculate_single_node_index(G):
    """
    Calculates the importance index (R) for each node.
    This index is the sum of the absolute differences of the Fiedler vector
    components between a node and its direct neighbors.
    """
    nodelist = sorted(G.nodes())
    if not nodelist:
        return {}, np.array([])
        
    L = nx.laplacian_matrix(G, nodelist=nodelist).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]
    
    indices = {}
    for node in nodelist:
        r_i = 0
        node_idx = nodelist.index(node)
        xi = fiedler_vector[node_idx]
        
        for neighbor in G.neighbors(node):
            neighbor_idx = nodelist.index(neighbor)
            xj = fiedler_vector[neighbor_idx]
            r_i += abs(xi - xj)
        indices[node] = r_i
    return indices, fiedler_vector

def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector):
    """Calculates the importance matrix (R2) for all pairs of nodes."""
    nodelist = sorted(G.nodes())
    num_nodes = len(nodelist)
    if num_nodes == 0:
        return np.array([[]])

    R2_matrix = np.zeros((num_nodes, num_nodes))
    fiedler_map = {node: fiedler_vector[i] for i, node in enumerate(nodelist)}

    for j_idx, node_j in enumerate(nodelist):
        for k_idx, node_k in enumerate(nodelist):
            if j_idx >= k_idx: continue
            
            R_j = single_node_indices[node_j]
            R_k = single_node_indices[node_k]
            
            correction_term = 0
            if G.has_edge(node_j, node_k):
                x_j = fiedler_map[node_j]
                x_k = fiedler_map[node_k]
                correction_term = abs(x_j - x_k)

            R_jk = R_j + R_k - correction_term
            R2_matrix[j_idx, k_idx] = R_jk
            
    return R2_matrix

# ---------- Main Execution Block ----------
if __name__ == "__main__" and 'G_edge' in locals() and G_edge.number_of_nodes() > 0:
    networks = {
        "Jazz Musicians": G_edge
    }
    results = []
    for name, G in networks.items():
        print(f"\n--- Calculating for {name} Network ---")
        single_indices, fiedler = calculate_single_node_index(G)
        R2 = calculate_node_pair_matrix(G, single_indices, fiedler)

        # =================================================================
        # THIS IS THE NEW SECTION THAT PRINTS THE INDEX FOR EVERY NODE
        # =================================================================
        print("\n--- Adjacent Edge Index (R-value) for All Nodes ---")
        
        # We loop through every item in the 'single_indices' dictionary
        # and print the node and its calculated index, formatted to 4 decimal places.
        for node_id, r_value in sorted(single_indices.items()):
            print(f"Node {node_id}: Index = {r_value:.4f}")
            
        print("-" * 50)
        # =================================================================

        # Find and display the most important node
        if single_indices:
            node_imp_item = max(single_indices.items(), key=lambda x: x[1])
            imp_node, imp_val = node_imp_item
            print(f"\nMost important node: {imp_node} (Index={imp_val:.4f})")
        else:
            imp_node, imp_val = "N/A", 0

        # Find and display the most important pair
        if R2.size > 0:
            idx = np.unravel_index(np.argmax(R2), R2.shape)
            nodelist = sorted(G.nodes())
            imp_pair = (nodelist[idx[0]], nodelist[idx[1]])
            imp_pair_val = R2[idx]
            print(f"Most important pair: {imp_pair} (Index={imp_pair_val:.4f})")
        else:
            imp_pair, imp_pair_val = "N/A", 0

        # The rest of your analysis can go here
        results.append([name, imp_node, imp_val, "N/A", imp_pair, imp_pair_val, "N/A"])

    # --- Create Summary Table ---
    df = pd.DataFrame(results, columns=["Network", "Most Imp Node", "R(node)", "SyncTime(node)",
                                        "Most Imp Pair", "R(pair)", "SyncTime(pair)"])
    print("\n--- Summary ---")
    print(df.to_string(index=False))