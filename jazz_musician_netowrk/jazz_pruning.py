import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import operator

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

    print("Initial Network Information:")
    print(f"Number of nodes: {G_edge.number_of_nodes()}")
    print(f"Number of edges: {G_edge.number_of_edges()}")
    print("-" * 20)

    # --- Visualize the initial network ---
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(G_edge)
    nx.draw(G_edge, pos, with_labels=True, node_color='skyblue', node_size=200, 
            edge_color='gray', font_size=8, font_weight='bold')
    plt.title("Initial Jazz Musicians Network Visualization", size=20)
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

# =================================================================
# THIS IS THE FUNCTION FOR PRUNING THE NETWORK AND EXPORTING
# =================================================================
def prune_network(G_original, pruning_percentage=10):
    """
    Iteratively prunes the network by removing the least important nodes.

    Args:
        G_original (nx.Graph): The original graph to be pruned.
        pruning_percentage (int): The percentage of nodes to remove.
    """
    G_pruned = G_original.copy()
    num_nodes_to_remove = int(G_pruned.number_of_nodes() * (pruning_percentage / 100))
    
    print(f"\n--- Starting Network Pruning (Removing {num_nodes_to_remove} nodes) ---")

    for i in range(num_nodes_to_remove):
        print(f"\n--- Pruning Iteration {i + 1} ---")
        
        # 1. Calculate the adjacent edge index for the current graph
        single_indices, _ = calculate_single_node_index(G_pruned)
        
        if not single_indices:
            print("Network is empty, stopping pruning.")
            break
            
        # 2. Identify the node with the lowest index
        least_important_node = min(single_indices.items(), key=operator.itemgetter(1))[0]
        
        print(f"Removing least important node: {least_important_node} (Index: {single_indices[least_important_node]:.4f})")
        
        # 3. Remove that node from the network
        G_pruned.remove_node(least_important_node)
        
        # 4. Store and print the new hierarchy
        new_indices, _ = calculate_single_node_index(G_pruned)
        if new_indices:
            sorted_nodes = sorted(new_indices.items(), key=operator.itemgetter(1), reverse=True)
            print("New hierarchy of most important nodes:")
            for node, index in sorted_nodes[:5]: # Print top 5
                print(f"  Node {node}: Index = {index:.4f}")
        else:
            print("No nodes left to rank.")
            
    print(f"\n--- Pruning Complete ---")
    print(f"Final number of nodes: {G_pruned.number_of_nodes()}")
    
    # --- Visualize the pruned network ---
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(G_pruned)
    nx.draw(G_pruned, pos, with_labels=True, node_color='lightcoral', node_size=200, 
            edge_color='gray', font_size=8, font_weight='bold')
    plt.title(f"Pruned Jazz Musicians Network ({pruning_percentage}% nodes removed)", size=20)
    plt.show()

    # =================================================================
    # THIS IS THE NEW SECTION THAT SAVES THE .GEXF FILE
    # =================================================================
    output_filename = "pruned_jazz_musicians_network.gexf"
    nx.write_gexf(G_pruned, output_filename)
    print(f"\nPruned network has been saved to '{output_filename}'")
    # =================================================================
    
    return G_pruned

# ---------- Main Execution Block ----------
if __name__ == "__main__" and 'G_edge' in locals() and G_edge.number_of_nodes() > 0:
    # --- Analysis of the Original Network ---
    print("\n--- Analysis of the Original Network ---")
    single_indices, fiedler = calculate_single_node_index(G_edge)
    R2 = calculate_node_pair_matrix(G_edge, single_indices, fiedler)

    print("\n--- Adjacent Edge Index (R-value) for All Nodes (Original Network) ---")
    for node_id, r_value in sorted(single_indices.items()):
        print(f"Node {node_id}: Index = {r_value:.4f}")
    print("-" * 50)
    
    if single_indices:
        imp_node = max(single_indices.items(), key=operator.itemgetter(1))[0]
        imp_val = single_indices[imp_node]
        print(f"\nMost important node: {imp_node} (Index={imp_val:.4f})")
    
    if R2.size > 0:
        idx = np.unravel_index(np.argmax(R2), R2.shape)
        nodelist = sorted(G_edge.nodes())
        imp_pair = (nodelist[idx[0]], nodelist[idx[1]])
        imp_pair_val = R2[idx]
        print(f"Most important pair: {imp_pair} (Index={imp_pair_val:.4f})")
        
    # --- Perform Pruning and Save the Result ---
    # You can change the percentage to whatever you want
    pruned_graph = prune_network(G_edge, pruning_percentage=20)