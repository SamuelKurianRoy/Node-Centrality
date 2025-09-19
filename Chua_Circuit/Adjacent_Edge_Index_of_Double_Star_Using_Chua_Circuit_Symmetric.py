import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- 1. Define the Double-Star Network ---
def create_double_star_network():
    G = nx.Graph()
    nodes = range(1, 16)
    G.add_nodes_from(nodes)

    # First star (hub: 2) connected to bridge (1)
    G.add_edge(1, 2)
    for i in range(3, 9):
        if i == 8:
            continue
        G.add_edge(2, i)

    # Second star (hub: 8) connected to bridge (1)
    G.add_edge(1, 8)
    for i in range(9, 16):
        G.add_edge(8, i)

    return G

# --- 2. Calculate the Adjacent Edge Index (for single nodes) with Chua States ---
def calculate_single_node_index(G, chua_states, time_idx):
    nodelist = sorted(G.nodes())
    N = G.number_of_nodes()
    
    # Extract Chua circuit states at the given time
    p_states = chua_states[time_idx, 0:N]
    q_states = chua_states[time_idx, N:2*N]
    r_states = chua_states[time_idx, 2*N:3*N]
    
    # Calculate the state vector magnitude for each node
    state_vectors = np.column_stack((p_states, q_states, r_states))
    
    indices = {}
    fiedler_vector = np.zeros(N)  # We'll use normalized state magnitudes instead
    
    # Calculate normalized state magnitudes
    state_magnitudes = np.linalg.norm(state_vectors, axis=1)
    state_magnitudes = (state_magnitudes - np.min(state_magnitudes)) / (np.max(state_magnitudes) - np.min(state_magnitudes))
    
    for i, node in enumerate(nodelist):
        r_i = 0
        xi = state_magnitudes[i]
        fiedler_vector[i] = xi  # Store normalized magnitude as Fiedler-like vector
        
        for neighbor in G.neighbors(node):
            j = nodelist.index(neighbor)
            xj = state_magnitudes[j]
            # Calculate difference in state magnitudes between connected nodes
            r_i += np.abs(xi - xj)
        
        indices[node] = r_i

    return indices, fiedler_vector

# --- 3. Calculate the Adjacent Edge Index Matrix (for node pairs) ---
def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector):
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

# --- 4. Chua’s nonlinear element ---
def g(p, gamma3=-4.0/3.0, gamma4=-3.0/4.0):
    return gamma4 * p + 0.5 * (gamma3 - gamma4) * (np.abs(p + 1) - np.abs(p - 1))

# --- 5. Networked Chua Dynamics ---
def chua_network_dynamics(t, state, G, gamma1=10.0, gamma2=18.0, kappa=0.05):
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G)

    p = state[0:N]
    q = state[N:2*N]
    r = state[2*N:3*N]

    dp = np.zeros(N)
    dq = np.zeros(N)
    dr = np.zeros(N)

    for i in range(N):
        dp[i] = -gamma1 * (p[i] - q[i] + g(p[i]))
        dq[i] = p[i] - q[i] + r[i]
        dr[i] = -gamma2 * q[i]

        # Diffusive coupling in p
        for j in range(N):
            if A[i, j] == 1:
                dp[i] += kappa * (p[j] - p[i])

    return np.concatenate([dp, dq, dr])

# --- 6. Network Visualization with Chua Circuit States ---
def plot_network_with_chua_states(G, states, time_idx, save_path=None):
    plt.figure(figsize=(15, 10))
    
    # Get the number of nodes
    N = G.number_of_nodes()
    
    # Extract states for the specific time index
    p_states = states[time_idx, 0:N]
    q_states = states[time_idx, N:2*N]
    r_states = states[time_idx, 2*N:3*N]
    
    # Create layout for the network
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Node colors based on p-state (voltage)
    node_colors = p_states
    
    # Draw the network
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nodes = nx.draw_networkx_nodes(G, pos, 
                                 node_color=node_colors,
                                 node_size=500,
                                 cmap=plt.cm.viridis,
                                 vmin=min(p_states),
                                 vmax=max(p_states))
    
    # Add colorbar
    plt.colorbar(nodes, label='Voltage (p-state)')
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title(f'Network State at t={time_idx}')
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# --- Plot Initial Network Structure ---
def plot_initial_network(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, seed=42)
    
    # Draw the network with different colors for different node types
    # Bridge node (1) in red
    nx.draw_networkx_nodes(G, pos, nodelist=[1], node_color='red', 
                          node_size=1000, label='Bridge Node')
    
    # Hub nodes (2 and 8) in blue
    nx.draw_networkx_nodes(G, pos, nodelist=[2, 8], node_color='blue', 
                          node_size=800, label='Hub Nodes')
    
    # Other nodes in green
    other_nodes = [n for n in G.nodes() if n not in [1, 2, 8]]
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='green',
                          node_size=500, label='Peripheral Nodes')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Add labels to nodes
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Initial Double-Star Network Structure", fontsize=16)
    plt.legend(fontsize=10)
    plt.axis('equal')
    plt.show()

# --- Main execution ---
if __name__ == '__main__':
    # Step 1: Build network
    G = create_double_star_network()
    
    # Plot the initial network structure
    print("Plotting initial network structure...")
    plot_initial_network(G)
    
    # Step 2: Set up initial conditions
    N = G.number_of_nodes()
    initial_state = np.random.uniform(-0.1, 0.1, 3*N)  # Random initial conditions
    
    print("Simulating Chua network dynamics...")
    
    # Step 3: Simulate the system
    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(
        chua_network_dynamics,
        t_span,
        initial_state,
        args=(G,),
        t_eval=t_eval,
        method='RK45'
    )
    
    # Step 4: Create output directory for plots if it doesn't exist
    output_dir = 'network_states'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 5: Plot network states at different time points
    time_points = [0, 250, 500, 750, 999]  # Plot at different time points
    for idx in time_points:
        save_path = os.path.join(output_dir, f'network_state_t{idx}.png')
        plot_network_with_chua_states(G, sol.y.T, idx, save_path)
    pos = nx.spring_layout(G, seed=42)

    # Calculate indices at different time points to see how they evolve
    analysis_times = [0, int(len(t_eval)/2), -1]  # Beginning, middle, and end of simulation
    for t_idx in analysis_times:
        print(f"\nAnalyzing network at time t = {t_eval[t_idx]:.2f}")
        
        # Step 2: Calculate node indices using Chua circuit states
        single_node_indices, state_vector = calculate_single_node_index(G, sol.y.T, t_idx)

    print("Adjacent Edge Index (R_i) for each node:")
    for node, index in single_node_indices.items():
        print(f"Node {node}: {index:.4f}")

    most_important_node = max(single_node_indices, key=single_node_indices.get)

    # Step 3: Node pair matrix
    R2_matrix = calculate_node_pair_matrix(G, single_node_indices, state_vector)
    nodelist = sorted(G.nodes())
    R2_df = pd.DataFrame(R2_matrix, index=nodelist, columns=nodelist)

    max_val = np.max(R2_matrix)
    max_pos = np.unravel_index(np.argmax(R2_matrix), R2_matrix.shape)
    max_row, max_col = nodelist[max_pos[0]], nodelist[max_pos[1]]

    most_important_pair = (max_row, max_col)

    print(f"\nMost Important Node: {most_important_node}")
    print(f"Most Important Pair: {most_important_pair} with R² = {max_val:.4f}")

    # Print R² matrix fully
    print("\nR² Matrix (for node pairs):")
    for i, row in enumerate(R2_df.values):
        row_str = []
        for j, val in enumerate(row):
            if (i, j) == max_pos:
                row_str.append(f"\033[1;32m{val:.4f}*\033[0m")  # highlight max
            else:
                row_str.append(f"{val:.4f}")
        print(f"{nodelist[i]:2d} [{', '.join(row_str)}]")

    # Step 4: Export to GEXF
    nx.set_node_attributes(G, single_node_indices, name="Ri")
    for u, v in G.edges():
        i = nodelist.index(u)
        j = nodelist.index(v)
        R2_val = R2_matrix[i, j] if R2_matrix[i, j] != 0 else R2_matrix[j, i]
        G[u][v]["R2"] = float(R2_val)

    file_name = "double_star_with_Ri_R2.gexf"
    nx.write_gexf(G, file_name)
    file_path = os.path.join(os.getcwd(), file_name)
    print(f"\nExported network to {file_name} at path: {file_path}")

    # Step 5: Simulate networked Chua dynamics
    N = G.number_of_nodes()
    initial_state = 0.1 * np.random.rand(3*N)
    t_span = [0, 50]
    t_eval = np.linspace(t_span[0], t_span[1], 5000)

    print("\nSimulating networked Chua circuits...")
    solution = solve_ivp(
        fun=lambda t, y: chua_network_dynamics(t, y, G),
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )

    # Extract dynamics of most important node
    idx = nodelist.index(most_important_node)
    p = solution.y[idx, :]
    q = solution.y[N+idx, :]
    r = solution.y[2*N+idx, :]

    # Plot attractor
    plt.figure(figsize=(8, 6))
    plt.plot(p, q, lw=0.5, color='black')
    plt.title(f"Chua Attractor of Node {most_important_node}", fontsize=14)
    plt.xlabel("p")
    plt.ylabel("q")
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p, q, r, lw=0.5, color='blue')
    ax.set_title(f"3D Attractor of Node {most_important_node}", fontsize=14)
    ax.set_xlabel("p")
    ax.set_ylabel("q")
    ax.set_zlabel("r")
    plt.show()
