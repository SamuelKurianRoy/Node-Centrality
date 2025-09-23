# -*- coding: utf-8 -*-
"""
Framework for a computational neuroscience experiment based on the research plan:
"Identifying Structural Bottlenecks for Inter-hemispheric Communication"

This script implements:
1.  A bi-hemispheric small-world network model ("brain").
2.  Topological analysis using the "adjacent edge index".
3.  A dynamic simulation using the Chialvo neuron model.
4.  A validation experiment via simulated lesions to test a cross-brain communication task.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- 1. Build the Brain Model ----------
def create_brain_model(nodes_per_hemisphere=100, k_intra=10, p_intra=0.1, n_corpus_callosum=10):
    """
    Creates a bi-hemispheric brain model.

    Args:
        nodes_per_hemisphere (int): Number of neurons in each hemisphere.
        k_intra (int): Each node is connected to k nearest neighbors in the ring topology.
        p_intra (int): The probability of rewiring each edge within a hemisphere.
        n_corpus_callosum (int): Number of connections between the two hemispheres.

    Returns:
        networkx.Graph: The complete brain network model.
    """
    # Create two separate small-world networks for the hemispheres
    h1 = nx.watts_strogatz_graph(nodes_per_hemisphere, k_intra, p_intra, seed=42)
    h2 = nx.watts_strogatz_graph(nodes_per_hemisphere, k_intra, p_intra, seed=10)

    # Relabel nodes in the second hemisphere to avoid overlap
    h2 = nx.relabel_nodes(h2, {i: i + nodes_per_hemisphere for i in h2.nodes()})

    # Combine the two hemispheres into a single graph
    brain_net = nx.compose(h1, h2)

    # Add corpus callosum connections between hemispheres
    h1_nodes = list(range(nodes_per_hemisphere))
    h2_nodes = list(range(nodes_per_hemisphere, 2 * nodes_per_hemisphere))
    
    # Select random nodes from each hemisphere to connect
    np.random.seed(42)
    source_nodes = np.random.choice(h1_nodes, size=n_corpus_callosum, replace=False)
    target_nodes = np.random.choice(h2_nodes, size=n_corpus_callosum, replace=False)

    for i in range(n_corpus_callosum):
        brain_net.add_edge(source_nodes[i], target_nodes[i])

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


# ---------- 3. Dynamic Experiment ----------
def run_communication_task(G, source_node, target_node, lesion_nodes=None, 
                           sim_steps=500, coupling_strength=0.1, stim_strength=0.15):
    """
    Runs the cross-brain communication task simulation.
    Uses the discrete-time Chialvo map for neuron dynamics.
    
    Args:
        G (networkx.Graph): The brain network.
        source_node (int): The neuron to stimulate.
        target_node (int): The neuron to monitor for signal arrival.
        lesion_nodes (list, optional): A list of nodes to remove/inhibit.
        sim_steps (int): Number of time steps to simulate.
        coupling_strength (float): Strength of connections between neurons.
        stim_strength (float): Strength of the initial stimulus.

    Returns:
        int: The time step of signal arrival at the target, or infinity if it doesn't arrive.
    """
    nodelist = sorted(G.nodes())
    N = G.number_of_nodes()
    
    # Create an adjacency matrix, handling lesions if any
    adj_matrix = nx.to_numpy_array(G, nodelist=nodelist)
    if lesion_nodes:
        for node_idx in lesion_nodes:
            adj_matrix[node_idx, :] = 0
            adj_matrix[:, node_idx] = 0

    # Chialvo map parameters
    a, b, c = 0.89, 0.6, 0.28
    
    # Neuron state variables (x: membrane potential, y: recovery variable)
    x = np.random.rand(N) * 0.1
    y = np.random.rand(N) * 0.1
    
    # Spike threshold
    spike_threshold = 0.5
    target_has_spiked = False
    arrival_time = np.inf

    # Simulation loop (discrete time steps)
    for t in range(sim_steps):
        # Calculate coupling input from neighbors
        # I_coupling = k * sum(A_ij * (x_j - x_i)) for each neuron i
        mean_neighbor_potential = (adj_matrix @ x) / (adj_matrix.sum(axis=1) + 1e-9)
        coupling_input = coupling_strength * (mean_neighbor_potential - x)
        
        # Define external input (stimulus)
        stimulus_input = np.zeros(N)
        if 10 <= t < 20: # Apply a brief stimulus pulse
            stimulus_input[source_node] = stim_strength
            
        total_input = coupling_input + stimulus_input
        
        # Update neuron states using the Chialvo map equations
        x_new = (x**2) * np.exp(y - x) + total_input
        y_new = a * y - b * x + c
        
        x, y = x_new, y_new
        
        # Check for signal arrival at the target
        if not target_has_spiked and x[target_node] > spike_threshold:
            arrival_time = t
            target_has_spiked = True
            break # End simulation once the signal arrives
            
    return arrival_time

# ---------- 4. Validation & 5. Analysis ----------
if __name__ == "__main__":
    # --- Setup ---
    N_PER_HEMISPHERE = 100
    TOTAL_NODES = 2 * N_PER_HEMISPHERE
    LESION_PERCENT = 5
    N_LESION = int(TOTAL_NODES * (LESION_PERCENT / 100))

    # --- Step 1: Build the Brain Model ---
    brain_network = create_brain_model(nodes_per_hemisphere=N_PER_HEMISPHERE)
     # Draw initial network
    pos = nx.spring_layout(brain_network, seed=42)
    nx.draw(brain_network, pos, with_labels=True, node_size=1000, node_color="skyblue", edge_color="gray")
    plt.title("Double-Star Network", fontsize=14)
    plt.show()
    # --- Step 2: Topological Prediction ---
    print("\nCalculating adjacent edge index for all neurons...")
    adj_indices = calculate_adjacent_edge_index(brain_network)
    
    # Sort nodes by their index to find the most "important" ones
    sorted_nodes_by_index = sorted(adj_indices.items(), key=lambda item: item[1], reverse=True)
    
    # Identify nodes for lesioning
    index_lesion_nodes = [node for node, index in sorted_nodes_by_index[:N_LESION]]
    all_nodes = list(brain_network.nodes())
    random_lesion_nodes = list(np.random.choice(
        [n for n in all_nodes if n not in index_lesion_nodes], 
        size=N_LESION, 
        replace=False
    ))
    
    print(f"Identified {N_LESION} nodes for lesioning based on the index.")
    print(f"Selected {N_LESION} random nodes for control lesioning.")

    # --- Step 3 & 4: Run Dynamic Experiment and Validation ---
    print("\nRunning simulations to test cross-brain communication...")
    
    # Define the communication task
    SOURCE_NEURON = 10  # A neuron in the left hemisphere
    TARGET_NEURON = 150 # A neuron in the right hemisphere

    # Run baseline simulation (intact brain)
    print("  - Simulating Baseline (no lesion)...")
    time_baseline = run_communication_task(brain_network, SOURCE_NEURON, TARGET_NEURON)

    # Run simulation with index-based lesion
    print("  - Simulating with Index-based Lesion...")
    time_index_lesion = run_communication_task(brain_network, SOURCE_NEURON, TARGET_NEURON, lesion_nodes=index_lesion_nodes)

    # Run simulation with random lesion
    print("  - Simulating with Random Lesion (control)...")
    time_random_lesion = run_communication_task(brain_network, SOURCE_NEURON, TARGET_NEURON, lesion_nodes=random_lesion_nodes)

    # --- Step 5: Analyze and Display Results ---
    print("\n" + "="*50)
    print("EXPERIMENTAL RESULTS: SIGNAL ARRIVAL TIME")
    print("="*50)
    print(f"Task: Signal from Neuron {SOURCE_NEURON} to Neuron {TARGET_NEURON}")
    print("-"*50)
    print(f"Baseline (Intact Brain): {time_baseline} steps")
    print(f"With Random Lesion:      {time_random_lesion} steps")
    print(f"With Index-based Lesion:  {time_index_lesion} steps")
    print("="*50)

    if time_index_lesion > time_random_lesion:
        print("\nConclusion: The hypothesis is supported!")
        print("Removing nodes with a high adjacent edge index caused a greater disruption")
        print("to communication than removing random nodes.")
    else:
        print("\nConclusion: The hypothesis was not supported in this run.")
        print("Removing nodes with a high index did not cause a greater disruption.")