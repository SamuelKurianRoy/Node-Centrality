import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import numpy as np 

# Define the path to your network file
file_path = 'Jazz-Musicians-Network.wl'

if os.path.exists(file_path):
    # Read the content of the file
    with open(file_path, 'r') as f:
        file_content = f.read()

    # Extract edges using regular expressions
    # This pattern looks for UndirectedEdge[node1, node2]
    edges = re.findall(r'UndirectedEdge\[(\d+),\s*(\d+)\]', file_content)

    # Create a graph from the extracted edges
    G_edge = nx.Graph()
    G_edge.add_edges_from([(int(u), int(v)) for u, v in edges])


    # --- Print some basic information about the graph ---
    print("Network Information:")
    print(f"Number of nodes: {G_edge.number_of_nodes()}")
    print(f"Number of edges: {G_edge.number_of_edges()}")
    print("-" * 20)


    # --- Visualize the network ---
    plt.figure(figsize=(10, 8)) # Set the figure size for better viewing

    # Use Kamada-Kawai layout for better visualization of larger graphs
    pos = nx.kamada_kawai_layout(G_edge)

    nx.draw(G_edge,
            pos,
            with_labels=True,
            node_color='skyblue',
            node_size=200, # Reduce node size
            edge_color='gray',
            font_size=10, # Reduce font size
            font_weight='bold')

    plt.title("Network Visualization from .wl File", size=20)
    plt.show()

else:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure your .wl file is in the same directory as this notebook or provide the full path.")



# ---------- Centrality calculations ----------
def calculate_single_node_index(G):
    nodelist = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]  # 2nd smallest
    indices = {}
    for i in nodelist:
        r_i = 0
        xi = fiedler_vector[i - 1]
        for neighbor in G.neighbors(i):
            xj = fiedler_vector[neighbor - 1]
            r_i += abs(xi - xj)
        indices[i] = r_i
    return indices, fiedler_vector

def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector):
    nodelist = sorted(G.nodes())
    num_nodes = len(nodelist)
    R2_matrix = np.zeros((num_nodes, num_nodes))
    for j_idx, node_j in enumerate(nodelist):
        for k_idx, node_k in enumerate(nodelist):
            if j_idx >= k_idx: continue
            R_j = single_node_indices[node_j]
            R_k = single_node_indices[node_k]
            intersection_count = 1 if G.has_edge(node_j, node_k) else 0
            x_j = fiedler_vector[j_idx]
            x_k = fiedler_vector[k_idx]
            correction_term = intersection_count * abs(x_j - x_k)
            R_jk = R_j + R_k - correction_term
            R2_matrix[j_idx, k_idx] = R_jk
    return R2_matrix


def calculate_single_node_index(G):
    nodelist = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]  # 2nd smallest
    indices = {}
    for i in nodelist:
        r_i = 0
        xi = fiedler_vector[i - 1]
        for neighbor in G.neighbors(i):
            xj = fiedler_vector[neighbor - 1]
            r_i += abs(xi - xj)
        indices[i] = r_i
    return indices, fiedler_vector

def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector):
    nodelist = sorted(G.nodes())
    num_nodes = len(nodelist)
    R2_matrix = np.zeros((num_nodes, num_nodes))
    for j_idx, node_j in enumerate(nodelist):
        for k_idx, node_k in enumerate(nodelist):
            if j_idx >= k_idx: continue
            R_j = single_node_indices[node_j]
            R_k = single_node_indices[node_k]
            intersection_count = 1 if G.has_edge(node_j, node_k) else 0
            x_j = fiedler_vector[j_idx]
            x_k = fiedler_vector[k_idx]
            correction_term = intersection_count * abs(x_j - x_k)
            R_jk = R_j + R_k - correction_term
            R2_matrix[j_idx, k_idx] = R_jk
    return R2_matrix

def add_centrality_attributes_to_graph(G, single_node_indices, fiedler_vector, R2_matrix, include_virtual_edges=True):
    """Add centrality metrics as node and edge attributes"""
    nodelist = sorted(G.nodes())
    
    # Add node attributes
    for i, node in enumerate(nodelist):
        G.nodes[node]['fiedler_value'] = float(fiedler_vector[i])
        G.nodes[node]['R_value'] = float(single_node_indices[node])
        G.nodes[node]['degree'] = G.degree(node)
        G.nodes[node]['node_id'] = node
        
        # Add normalized versions for better visualization in Gephi
        G.nodes[node]['fiedler_normalized'] = float(fiedler_vector[i])
        G.nodes[node]['R_normalized'] = float(single_node_indices[node])
    
    # Store original edges for marking
    original_edges = set(G.edges())
    
    # # Add virtual edges for all non-connected pairs if requested
    # if include_virtual_edges:
    #     for i in range(len(nodelist)):
    #         for j in range(i+1, len(nodelist)):
    #             node_u, node_v = nodelist[i], nodelist[j]
    #             if not G.has_edge(node_u, node_v):
    #                 G.add_edge(node_u, node_v)
    
    # Add edge attributes with R² values for ALL edges (original + virtual)
    for edge in G.edges():
        node_u, node_v = edge
        u_idx = nodelist.index(node_u)
        v_idx = nodelist.index(node_v)
        
        # Get R² value for this pair
        if u_idx < v_idx:
            r2_value = R2_matrix[u_idx, v_idx]
        else:
            r2_value = R2_matrix[v_idx, u_idx]
        
        G.edges[edge]['R2_value'] = float(r2_value)
        
        # Mark edge type
        if edge in original_edges or (node_v, node_u) in original_edges:
            G.edges[edge]['edge_type'] = 'original'
            G.edges[edge]['weight'] = 1.0  # Original edges have weight 1
        else:
            G.edges[edge]['edge_type'] = 'virtual'
            G.edges[edge]['weight'] = 0.1  # Virtual edges have lower weight
        
        # Add individual R values for the connected nodes
        G.edges[edge]['R_u'] = float(single_node_indices[node_u])
        G.edges[edge]['R_v'] = float(single_node_indices[node_v])
        
        # Add Fiedler difference
        u_fiedler = fiedler_vector[u_idx]
        v_fiedler = fiedler_vector[v_idx]
        G.edges[edge]['fiedler_diff'] = float(abs(u_fiedler - v_fiedler))
        
        # Add sum of individual R values for comparison
        G.edges[edge]['R_sum'] = float(single_node_indices[node_u] + single_node_indices[node_v])

def export_enhanced_gexf(network_name, G):
    """Export graph with all centrality attributes to GEXF"""
    # Calculate centrality metrics
    single_indices, fiedler = calculate_single_node_index(G)
    R2_matrix = calculate_node_pair_matrix(G, single_indices, fiedler)
    
    # Add attributes to graph
    add_centrality_attributes_to_graph(G, single_indices, fiedler, R2_matrix)
    
    # Normalize values for better visualization (0-1 range)
    all_r_values = list(single_indices.values())
    all_fiedler_values = list(fiedler)
    
    r_min, r_max = min(all_r_values), max(all_r_values)
    f_min, f_max = min(all_fiedler_values), max(all_fiedler_values)
    
    for node in G.nodes():
        if r_max != r_min:
            G.nodes[node]['R_normalized'] = (single_indices[node] - r_min) / (r_max - r_min)
        if f_max != f_min:
            node_idx = sorted(G.nodes()).index(node)
            G.nodes[node]['fiedler_normalized'] = (fiedler[node_idx] - f_min) / (f_max - f_min)
    
    # Export to GEXF
    filename = f"{network_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.gexf"
    nx.write_gexf(G, filename)
    print(f"Enhanced GEXF exported to: {filename}")
    
    # Print summary of attributes added
    print(f"Node attributes added: {list(G.nodes(data=True))[0][1].keys()}")
    print(f"Edge attributes added: {list(G.edges(data=True))[0][2].keys()}")
    
    return filename

# ---------- Main ----------
if __name__=="__main__":
    networks = {
        "Jazz Musicians": G_edge
    }
    results = []
    for name,G in networks.items():
        print(f"\n--- {name} ---")
        single_indices,fiedler = calculate_single_node_index(G)
        R2 = calculate_node_pair_matrix(G, single_indices, fiedler)

        # most important node
        node_imp = max(single_indices.items(), key=lambda x:x[1])
        imp_node, imp_val = node_imp
        print(f"Most important node: {imp_node} (R={imp_val:.4f})")

        # most important pair
        idx = np.unravel_index(np.argmax(R2), R2.shape)
        nodelist = sorted(G.nodes())
        imp_pair = (nodelist[idx[0]], nodelist[idx[1]])
        imp_pair_val = R2[idx]
        print(f"Most important pair: {imp_pair} (R={imp_pair_val:.4f})")

        # # run simulations
        # t_eval = np.linspace(0,12,600)
        # sol_node,_ = run_scenario(G,[imp_node],t_eval=t_eval)
        # err_node = sync_errors_from_solution(sol_node,len(G))
        # t_sync_node = time_to_sync(err_node,t_eval)

        # sol_pair,_ = run_scenario(G,list(imp_pair),t_eval=t_eval)
        # err_pair = sync_errors_from_solution(sol_pair,len(G))
        # t_sync_pair = time_to_sync(err_pair,t_eval)

        # print(f"Sync time (node {imp_node}): {t_sync_node:.2f}")
        # print(f"Sync time (pair {imp_pair}): {t_sync_pair:.2f}")

        # results.append([name, imp_node, imp_val, t_sync_node, imp_pair, imp_pair_val, t_sync_pair])

    # make summary table
    import pandas as pd
    df = pd.DataFrame(results, columns=["Network","Most Imp Node","R(node)","SyncTime(node)",
                                        "Most Imp Pair","R(pair)","SyncTime(pair)"])
    print("\nSummary:")
    print(df.to_string(index=False))

    # # G = create_double_star_13()
    # pos = nx.spring_layout(G, seed=42)

    # # Draw initial network
    # nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", edge_color="gray")
    # plt.title("Double-Star Network", fontsize=14)
    # plt.show()

    export_enhanced_gexf("Jazz_Musician", G)

    # nx.write_gexf(G, "double_star_network.gexf")