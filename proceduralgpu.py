import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh as cpu_eigsh

# Try to import CuPy for GPU acceleration
GPU_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cuda_spla
    print("CuPy found. GPU acceleration is available.")
    GPU_AVAILABLE = True
except ImportError:
    print("CuPy not found. Running on CPU only.")

# ==============================================================================
# PROCEDURAL NETWORK GENERATION (Unchanged)
# ==============================================================================
def generate_fractal_network(seed_graph, iterations):
    if iterations == 0:
        print("--- Generating Network: 0 iterations (copy of original) ---")
        return seed_graph.copy()

    print(f"--- Generating Network: {iterations} iteration(s) ---")
    
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
            
        for u, v in current_graph.edges():
            new_graph.add_edge((u, hub_node), (v, hub_node))
            
        current_graph = new_graph
        print(f"Iteration {i + 1} complete. New graph has {current_graph.number_of_nodes()} nodes.")
        
    final_graph = nx.convert_node_labels_to_integers(current_graph, first_label=1)
    
    return final_graph

# ==============================================================================
# ANALYSIS CODE
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

def calculate_single_node_index(G, use_gpu=False):
    if not nx.is_connected(G):
        print("Warning: Graph is not connected. Analyzing the largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc).copy()
        return calculate_single_node_index(subgraph, use_gpu=use_gpu)

    nodelist = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist)
    
    if use_gpu and GPU_AVAILABLE:
        print("Using CuPy for GPU-accelerated eigenvalue decomposition...")
        L_gpu = cp.sparse.csc_matrix(L.astype(float))
        eigenvalues, eigenvectors = cuda_spla.eigsh(L_gpu, k=2, which='SA')
        eigenvalues = eigenvalues.get()
        eigenvectors = eigenvectors.get()
    else:
        print("Using SciPy for CPU-based eigenvalue decomposition...")
        eigenvalues, eigenvectors = cpu_eigsh(L.astype(float), k=2, which='SA')

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    fiedler_vector = np.real(eigenvectors[:, order[1]])

    indices = {}
    for i, node_id in enumerate(nodelist):
        r_i = 0
        xi = fiedler_vector[i]
        for neighbor in G.neighbors(node_id):
            neighbor_idx = nodelist.index(neighbor)
            xj = fiedler_vector[neighbor_idx]
            r_i += abs(xi - xj)
        indices[node_id] = r_i
        
    return indices, fiedler_vector, nodelist

def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector, nodelist):
    full_nodelist_sorted = sorted(G.nodes())
    num_nodes = len(full_nodelist_sorted)
    R2_matrix = np.zeros((num_nodes, num_nodes))
    fiedler_map = {node: fv_val for node, fv_val in zip(nodelist, fiedler_vector)}
    
    for j_idx, node_j in enumerate(full_nodelist_sorted):
        for k_idx, node_k in enumerate(full_nodelist_sorted):
            if j_idx >= k_idx: continue
            R_j = single_node_indices.get(node_j, 0)
            R_k = single_node_indices.get(node_k, 0)
            intersection_count = 1 if G.has_edge(node_j, node_k) else 0
            x_j = fiedler_map.get(node_j, 0)
            x_k = fiedler_map.get(node_k, 0)
            correction_term = intersection_count * abs(x_j - x_k)
            R_jk = R_j + R_k - correction_term
            R2_matrix[j_idx, k_idx] = R_jk
    return R2_matrix

# ==============================================================================
# [NEW] GEXF EXPORT FUNCTIONALITY
# ==============================================================================

def export_enhanced_gexf(network_name, G, mapped_single_indices, mapped_fiedler_vector, R2_matrix):
    """
    Adds centrality metrics as attributes to a graph and exports it as a GEXF file.
    """
    print(f"\nAdding attributes and exporting '{network_name}' to GEXF...")
    
    full_nodelist_sorted = sorted(G.nodes())
    
    # 1. Add Node Attributes
    for i, node in enumerate(full_nodelist_sorted):
        G.nodes[node]['R_value'] = float(mapped_single_indices.get(node, 0))
        G.nodes[node]['fiedler_value'] = float(mapped_fiedler_vector[i])
        G.nodes[node]['degree'] = G.degree(node)

    # 2. Add Edge Attributes
    for u, v in G.edges():
        u_idx = full_nodelist_sorted.index(u)
        v_idx = full_nodelist_sorted.index(v)
        
        # Ensure indices are ordered correctly for accessing the upper triangle of R2_matrix
        if u_idx < v_idx:
            r2_val = R2_matrix[u_idx, v_idx]
        else:
            r2_val = R2_matrix[v_idx, u_idx]
        
        G.edges[u, v]['R2_value'] = float(r2_val)

    # 3. Normalize Attributes for Visualization (e.g., in Gephi)
    all_r_values = list(nx.get_node_attributes(G, 'R_value').values())
    all_fiedler_values = list(nx.get_node_attributes(G, 'fiedler_value').values())
    
    r_min, r_max = min(all_r_values), max(all_r_values)
    f_min, f_max = min(all_fiedler_values), max(all_fiedler_values)
    
    for node in G.nodes():
        if (r_max - r_min) > 0:
            G.nodes[node]['R_normalized'] = (G.nodes[node]['R_value'] - r_min) / (r_max - r_min)
        else:
            G.nodes[node]['R_normalized'] = 0.5
            
        if (f_max - f_min) > 0:
            G.nodes[node]['fiedler_normalized'] = (G.nodes[node]['fiedler_value'] - f_min) / (f_max - f_min)
        else:
            G.nodes[node]['fiedler_normalized'] = 0.5

    # 4. Write to File
    filename = f"{network_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.gexf"
    nx.write_gexf(G, filename)
    print(f"Enhanced GEXF file saved to: {filename}")
    
# ==============================================================================
# [MODIFIED] ANALYSIS AND MAIN EXECUTION
# ==============================================================================

def analyze_and_report(G, network_name, use_gpu_flag=False):
    print("\n" + "="*50)
    print(f"ANALYZING: {network_name}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("="*50)

    if G.number_of_nodes() == 0:
        print("Graph is empty, skipping analysis.")
        return None

    print("Calculating Fiedler vector for large sparse matrix... (This may take a moment)")
    # These results may be from a subgraph (the largest connected component)
    single_indices, fiedler_vector, subgraph_nodelist = calculate_single_node_index(G, use_gpu=use_gpu_flag)
    print("Calculation complete.")

    # Map the subgraph results to the full graph's node list
    full_nodelist_sorted = sorted(G.nodes())
    mapped_fiedler = np.zeros(len(full_nodelist_sorted))
    mapped_single_indices = {node: 0.0 for node in full_nodelist_sorted}

    for i, node_id in enumerate(subgraph_nodelist):
        if node_id in mapped_single_indices:
            mapped_single_indices[node_id] = single_indices[node_id]
            # Place the fiedler value in the correct position corresponding to the full sorted list
            full_list_idx = full_nodelist_sorted.index(node_id)
            mapped_fiedler[full_list_idx] = fiedler_vector[i]
            
    # Calculate R2 matrix using the full mapped data
    R2 = calculate_node_pair_matrix(G, mapped_single_indices, mapped_fiedler, full_nodelist_sorted)

    if not mapped_single_indices or all(v == 0 for v in mapped_single_indices.values()):
        print("No centrality indices calculated, cannot determine most important node.")
        return None

    node_imp = max(mapped_single_indices.items(), key=lambda x: x[1])
    imp_node, imp_val = node_imp
    print(f"Most important node: {imp_node} (R={imp_val:.4f})")

    idx = np.unravel_index(np.argmax(R2), R2.shape)
    imp_pair = (full_nodelist_sorted[idx[0]], full_nodelist_sorted[idx[1]])
    imp_pair_val = R2[idx]
    print(f"Most important pair: {imp_pair} (R={imp_pair_val:.4f})")
    
    # [MODIFIED] Return all calculated data needed for both the summary and the GEXF export
    summary_data = [network_name, imp_node, imp_val, imp_pair, imp_pair_val]
    export_data = {
        "mapped_indices": mapped_single_indices,
        "mapped_fiedler": mapped_fiedler,
        "R2_matrix": R2
    }
    return summary_data, export_data

if __name__=="__main__":
    file_path = 'Jazz-Musicians-Network.wl'
    G_seed = load_seed_network(file_path)

    if G_seed:
        plt.figure(figsize=(10, 8))
        pos = nx.kamada_kawai_layout(G_seed)
        nx.draw(G_seed, pos, with_labels=True, node_color='skyblue', node_size=200, edge_color='gray', font_size=10)
        plt.title("Original Seed Network (Jazz Musicians)", size=20)
        plt.show()

        USE_GPU_ACCELERATION = True and GPU_AVAILABLE 
        ITERATIONS_TO_RUN = [0, 1] 
        results_summary = []
        
        for i in ITERATIONS_TO_RUN:
            G_generated = generate_fractal_network(G_seed, iterations=i)
            network_name = f"Fractal Jazz Network ({i} iter)"
            
            analysis_output = analyze_and_report(G_generated, network_name, use_gpu_flag=USE_GPU_ACCELERATION)
            
            if analysis_output:
                summary_data, export_data = analysis_output
                results_summary.append(summary_data)
                
                # [NEW] Call the export function with the generated graph and its analysis data
                export_enhanced_gexf(
                    network_name,
                    G_generated,
                    export_data["mapped_indices"],
                    export_data["mapped_fiedler"],
                    export_data["R2_matrix"]
                )
        
        df = pd.DataFrame(results_summary, columns=["Network", "Most Imp Node", "R(node)", "Most Imp Pair", "R(pair)"])
        print("\n" + "="*50)
        print("SUMMARY OF RESULTS")
        print("="*50)
        print(df.to_string(index=False))
