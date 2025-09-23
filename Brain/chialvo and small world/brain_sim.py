# -*- coding: utf-8 -*-
"""
Final, stabilized framework for the computational neuroscience experiment.

This version corrects the numerical instability by implementing a more robust
diffusive coupling mechanism while retaining all advanced features.
"""
import numpy as np
import networkx as nx

# ---------- 1. Build the Brain Model (with E/I Connections) ----------
def create_brain_model(nodes_per_hemisphere=100, k_intra=10, p_intra=0.1, 
                       n_corpus_callosum=10, inhibitory_fraction=0.2):
    """
    Creates a bi-hemispheric model with both excitatory and inhibitory connections.
    """
    h1 = nx.watts_strogatz_graph(nodes_per_hemisphere, k_intra, p_intra, seed=42)
    h2 = nx.watts_strogatz_graph(nodes_per_hemisphere, k_intra, p_intra, seed=10)
    h2 = nx.relabel_nodes(h2, {i: i + nodes_per_hemisphere for i in h2.nodes()})
    brain_net = nx.compose(h1, h2)

    h1_nodes = list(range(nodes_per_hemisphere))
    h2_nodes = list(range(nodes_per_hemisphere, 2 * nodes_per_hemisphere))
    np.random.seed(42)
    source_nodes = np.random.choice(h1_nodes, size=n_corpus_callosum, replace=False)
    target_nodes = np.random.choice(h2_nodes, size=n_corpus_callosum, replace=False)
    for i in range(n_corpus_callosum):
        brain_net.add_edge(source_nodes[i], target_nodes[i])
        
    for edge in brain_net.edges():
        if np.random.rand() < inhibitory_fraction:
            brain_net.edges[edge]['type'] = 'inhibitory'
        else:
            brain_net.edges[edge]['type'] = 'excitatory'

    print(f"Created brain model with {brain_net.number_of_nodes()} neurons and {brain_net.number_of_edges()} connections.")
    return brain_net


# ---------- 2. Topological Prediction ----------
def calculate_adjacent_edge_index(G):
    """Calculates the adjacent edge index (R) for each node."""
    nodelist = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]
    
    indices = {}
    for i, node in enumerate(nodelist):
        r_i = 0
        xi = fiedler_vector[i]
        for neighbor in G.neighbors(node):
            neighbor_idx = nodelist.index(neighbor)
            xj = fiedler_vector[neighbor_idx]
            r_i += abs(xi - xj)
        indices[node] = r_i
    return indices


# ---------- GEXF Export for Gephi ----------
def export_to_gexf(G, indices, nodes_per_hemisphere, filename="final_brain_model.gexf"):
    """Exports the network with rich attributes to a GEXF file."""
    for node in G.nodes():
        G.nodes[node]['hemisphere'] = 1 if node < nodes_per_hemisphere else 2
        G.nodes[node]['adj_edge_index'] = float(indices.get(node, 0.0))

    nx.write_gexf(G, filename)
    print(f"\nNetwork model with attributes saved to '{filename}'.")


# ---------- 3. Dynamic Experiment (Final, Stable Version) ----------
def run_final_communication_task(G, source_node, target_node, lesion_nodes=None, 
                                 sim_steps=1000, coupling_strength=0.05, stim_strength=0.3,
                                 base_b=0.22, b_mismatch_std=0.01, noise_intensity=0.001):
    """
    Runs the simulation with a stable diffusive coupling mechanism.
    """
    nodelist = sorted(G.nodes())
    N = G.number_of_nodes()
    
    # --- New: Create separate adjacency matrices for E/I for stable coupling ---
    adj_excitatory = nx.to_numpy_array(G, nodelist=nodelist, weight=None)
    adj_inhibitory = np.zeros_like(adj_excitatory)
    
    for u, v, data in G.edges(data=True):
        u_idx, v_idx = nodelist.index(u), nodelist.index(v)
        if data['type'] == 'inhibitory':
            adj_inhibitory[u_idx, v_idx] = 1
            adj_inhibitory[v_idx, u_idx] = 1
            adj_excitatory[u_idx, v_idx] = 0 # Remove from excitatory matrix
            adj_excitatory[v_idx, u_idx] = 0

    if lesion_nodes:
        for node_idx in lesion_nodes:
            adj_excitatory[node_idx, :] = 0; adj_excitatory[:, node_idx] = 0
            adj_inhibitory[node_idx, :] = 0; adj_inhibitory[:, node_idx] = 0

    # Chialvo map parameters and neuron diversity
    a, c = 0.89, 0.28
    np.random.seed(42)
    b_params = np.random.normal(loc=base_b, scale=b_mismatch_std, size=N)
    
    x = np.random.rand(N) * 0.1
    y = np.random.rand(N) * 0.1
    
    spike_threshold = 0.5
    arrival_time = np.inf

    # Simulation loop
    for t in range(sim_steps):
        # --- New: Stable Diffusive Coupling Calculation ---
        # Excitatory coupling pulls neurons towards their neighbors' potential
        excitatory_sum = adj_excitatory @ x
        excitatory_degree = adj_excitatory.sum(axis=1)
        excitatory_input = coupling_strength * (excitatory_sum - excitatory_degree * x)
        
        # Inhibitory coupling provides a negative input based on neighbors' activity
        inhibitory_input = -coupling_strength * (adj_inhibitory @ x)
        
        coupling_input = excitatory_input + inhibitory_input
        
        # Stochastic Noise and Stimulus
        noise_input = noise_intensity * np.random.randn(N)
        stimulus_input = np.zeros(N)
        if 10 <= t < 20:
            stimulus_input[source_node] = stim_strength
            
        total_input = coupling_input + stimulus_input + noise_input
        
        # Update neuron states using the Chialvo map
        x_new = (x**2) * np.exp(y - x) + total_input
        y_new = a * y - b_params * x + c
        
        # --- New: Clipping to prevent explosion from random fluctuations ---
        x = np.clip(x_new, -1, 5)
        y = np.clip(y_new, -1, 5)
        
        if t > 20 and x[target_node] > spike_threshold:
            arrival_time = t
            break
            
    return arrival_time


# ---------- 4. Validation & 5. Analysis ----------
if __name__ == "__main__":
    # --- Setup & TUNABLE PARAMETERS ---
    COUPLING_K = 0.08
    STIMULUS_I = 0.35
    NOISE_EPSILON = 0.001
    N_PER_HEMISPHERE = 100
    LESION_PERCENT = 5

    # --- Step 1, 2: Build, Analyze, Export ---
    brain_network = create_brain_model(nodes_per_hemisphere=N_PER_HEMISPHERE)
    adj_indices = calculate_adjacent_edge_index(brain_network)
    export_to_gexf(brain_network, adj_indices, N_PER_HEMISPHERE)
    
    # Identify nodes for lesioning
    TOTAL_NODES = 2 * N_PER_HEMISPHERE
    N_LESION = int(TOTAL_NODES * (LESION_PERCENT / 100))
    sorted_nodes_by_index = sorted(adj_indices.items(), key=lambda item: item[1], reverse=True)
    index_lesion_nodes = [node for node, index in sorted_nodes_by_index[:N_LESION]]
    all_nodes = list(brain_network.nodes())
    random_lesion_nodes = list(np.random.choice([n for n in all_nodes if n not in index_lesion_nodes], size=N_LESION, replace=False))
    
    # --- Step 3, 4: Run Simulations ---
    print("\nRunning STABLE simulations to test cross-brain communication...")
    SOURCE_NEURON, TARGET_NEURON = 10, 150
    sim_params = {'coupling_strength': COUPLING_K, 'stim_strength': STIMULUS_I, 'noise_intensity': NOISE_EPSILON}

    time_baseline = run_final_communication_task(brain_network, SOURCE_NEURON, TARGET_NEURON, **sim_params)
    time_index_lesion = run_final_communication_task(brain_network, SOURCE_NEURON, TARGET_NEURON, lesion_nodes=index_lesion_nodes, **sim_params)
    time_random_lesion = run_final_communication_task(brain_network, SOURCE_NEURON, TARGET_NEURON, lesion_nodes=random_lesion_nodes, **sim_params)

    # --- Step 5: Display Results ---
    print("\n" + "="*50 + "\nEXPERIMENTAL RESULTS: SIGNAL ARRIVAL TIME\n" + "="*50)
    print(f"Task: Signal from Neuron {SOURCE_NEURON} to Neuron {TARGET_NEURON}\n" + "-"*50)
    print(f"Baseline (Intact Brain): {time_baseline} steps")
    print(f"With Random Lesion:      {time_random_lesion} steps")
    print(f"With Index-based Lesion:  {time_index_lesion} steps\n" + "="*50)

    if time_baseline == np.inf:
        print("\nConclusion: Signal did not propagate. Try increasing COUPLING_K or STIMULUS_I.")
    elif time_index_lesion > time_random_lesion:
        print("\nConclusion: The hypothesis is supported!")
    else:
        print("\nConclusion: The hypothesis was not supported in this run.")