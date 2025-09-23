import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
# [CHANGE] Import the sparse eigenvalue solver from SciPy
from scipy.sparse.linalg import eigs

# ==============================================================================
# [UPDATED] PROCEDURAL NETWORK GENERATION
# ==============================================================================

def generate_fractal_network(seed_graph, iterations):
    """
    Generates a larger, self-similar network using a hub-to-hub connection rule.
    """
    if iterations == 0:
        print("--- Generating Network: 0 iterations (copy of original) ---")
        return seed_graph.copy()

    print(f"--- Generating Network: {iterations} iteration(s) ---")
    
    # [CHANGE] Find the hub of the seed graph (highest degree node) to use for connections.
    hub_node = max(dict(seed_graph.degree()).items(), key=lambda x: x[1])[0]
    print(f"Seed graph hub node identified: {hub_node} (degree {seed_graph.degree[hub_node]})")

    current_graph = seed_graph.copy()
    
    for i in range(iterations):
        print(f"Starting iteration {i + 1}...")
        new_graph = nx.Graph()
        
        for node in current_graph.nodes():
            g_copy = nx.relabel_nodes(seed_graph, lambda x: (node, x))
            new_graph.add_nodes_from(g_copy.nodes())
            new_graph.add_edges_from(g_copy.edges())
            
        # [CHANGE] Use the more robust hub-to-hub connection rule.
        for u, v in current_graph.edges():
            # Connect the hub of cluster 'u' to the hub of cluster 'v'.
            new_graph.add_edge((u, hub_node), (v, hub_node))
            
        current_graph = new_graph
        print(f"Iteration {i + 1} complete. New graph has {current_graph.number_of_nodes()} nodes.")
        
    final_graph = nx.convert_node_labels_to_integers(current_graph, first_label=1)
    
    return final_graph

# ==============================================================================
# ORIGINAL CODE (with necessary adjustments)
# ==============================================================================

def load_seed_network(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            file_content = f.read()
        edges = re.findall(r'UndirectedEdge\[(\d+),\s*(\d+)\]', file_content)
        G = nx.Graph()
        G.add_edges_from([(int(u), int(v)) for u, v in edges])
        return G
    else:
        print(f"Error: The file '{file_path}' was not found.")
        return None

# [UPDATED] Centrality calculations to handle large, sparse matrices
def calculate_single_node_index(G):
    if not nx.is_connected(G):
        print("Warning: Graph is not connected. Analyzing the largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc).copy() # Use a copy to avoid modifying the subgraph
        return calculate_single_node_index(subgraph)

    nodelist = sorted(G.nodes())
    
    # [CHANGE] Keep the Laplacian as a sparse matrix. Do NOT use .toarray()
    L = nx.laplacian_matrix(G, nodelist=nodelist)
    
    # [CHANGE] Use the sparse solver. We ask for the 2 smallest magnitude ('SM') eigenvalues.
    # We use L.astype(float) to ensure the right data type for the solver.
    # k=2 because we want the one associated with eigenvalue 0 and the Fiedler vector.
    eigenvalues, eigenvectors = eigs(L.astype(float), k=2, which='SM')
    
    # [CHANGE] The results need to be sorted and processed correctly.
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # The Fiedler vector is the eigenvector of the second smallest eigenvalue.
    fiedler_vector = np.real(eigenvectors[:, 1])

    indices = {}
    for i, node_id in enumerate(nodelist):
        r_i = 0
        xi = fiedler_vector[i]
        for neighbor in G.neighbors(node_id):
            neighbor_idx = nodelist.index(neighbor)
            xj = fiedler_vector[neighbor_idx]
            r_i += abs(xi - xj)
        indices[node_id] = r_i
        
    # Also return the full graph's nodelist for mapping results back if we used a subgraph
    return indices, fiedler_vector, nodelist

# Your other functions remain mostly the same
def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector, nodelist):
    num_nodes = len(nodelist)
    R2_matrix = np.zeros((num_nodes, num_nodes))
    for j_idx, node_j in enumerate(nodelist):
        for k_idx, node_k in enumerate(nodelist):
            if j_idx >= k_idx: continue
            # Handle cases where a node might not be in single_node_indices (if graph was disconnected)
            R_j = single_node_indices.get(node_j, 0)
            R_k = single_node_indices.get(node_k, 0)

            intersection_count = 1 if G.has_edge(node_j, node_k) else 0
            x_j = fiedler_vector[j_idx]
            x_k = fiedler_vector[k_idx]
            correction_term = intersection_count * abs(x_j - x_k)
            R_jk = R_j + R_k - correction_term
            R2_matrix[j_idx, k_idx] = R_jk
    return R2_matrix

def analyze_and_report(G, network_name):
    print("\n" + "="*50)
    print(f"ANALYZING: {network_name}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("="*50)

    # Note: Calculation will be slower now, but will not crash.
    print("Calculating Fiedler vector for large sparse matrix... (This may take a moment)")
    single_indices, fiedler, nodelist = calculate_single_node_index(G)
    print("Calculation complete.")

    # The nodelist might be from the largest_cc, so we need to be careful
    original_nodelist = sorted(G.nodes())
    
    # Map fiedler vector and indices to the full graph if we used a subgraph
    full_fiedler = np.zeros(len(original_nodelist))
    full_indices = {}
    
    if len(nodelist) != len(original_nodelist):
        for i, node_id in enumerate(nodelist):
            orig_idx = original_nodelist.index(node_id)
            full_fiedler[orig_idx] = fiedler[i]
            full_indices[node_id] = single_indices[node_id]
    else: # Graph was connected, no mapping needed
        full_fiedler = fiedler
        full_indices = single_indices

    R2 = calculate_node_pair_matrix(G, full_indices, full_fiedler, original_nodelist)

    node_imp = max(full_indices.items(), key=lambda x: x[1])
    imp_node, imp_val = node_imp
    print(f"Most important node: {imp_node} (R={imp_val:.4f})")

    idx = np.unravel_index(np.argmax(R2), R2.shape)
    imp_pair = (original_nodelist[idx[0]], original_nodelist[idx[1]])
    imp_pair_val = R2[idx]
    print(f"Most important pair: {imp_pair} (R={imp_pair_val:.4f})")
    
    return [network_name, imp_node, imp_val, imp_pair, imp_pair_val]

# ---------- Main Execution Block (Unchanged) ----------
if __name__=="__main__":
    file_path = 'Jazz-Musicians-Network.wl'
    G_seed = load_seed_network(file_path)

    if G_seed:
        # Visualize only the small seed network
        plt.figure(figsize=(10, 8))
        pos = nx.kamada_kawai_layout(G_seed)
        nx.draw(G_seed, pos, with_labels=True, node_color='skyblue', node_size=200, edge_color='gray', font_size=10)
        plt.title("Original Seed Network (Jazz Musicians)", size=20)
        plt.show()

        ITERATIONS_TO_RUN = [0, 1] 
        results = []
        
        for i in ITERATIONS_TO_RUN:
            G_generated = generate_fractal_network(G_seed, iterations=i)
            network_name = f"Fractal Jazz Network ({i} iter)"
            analysis_results = analyze_and_report(G_generated, network_name)
            if analysis_results:
                results.append(analysis_results)
        
        df = pd.DataFrame(results, columns=["Network", "Most Imp Node", "R(node)", "Most Imp Pair", "R(pair)"])
        print("\n" + "="*50)
        print("SUMMARY OF RESULTS")
        print("="*50)
        print(df.to_string(index=False))