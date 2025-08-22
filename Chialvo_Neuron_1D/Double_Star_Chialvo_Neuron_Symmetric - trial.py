import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# --- 1D Chialvo Map (discrete neuron model) ---
def chialvo_1d_step(x, alpha=3.7, beta=0.01):
    """One iteration of the 1D Chialvo-like map."""
    return alpha * x * (1 - x) - beta

# --- Double-Star Network ---
def create_double_star_network():
    """
    Creates the 13-node double-star network.
    Node 1 is the bridge, nodes 2 and 8 are hubs.
    """
    G = nx.Graph()
    nodes = range(1, 14)
    G.add_nodes_from(nodes)

    # First star (hub: 2) connected to bridge (1)
    G.add_edge(1, 2)
    for i in range(3, 8):
        G.add_edge(2, i)

    # Second star (hub: 8) connected to bridge (1)
    G.add_edge(1, 8)
    for i in range(9, 14):
        G.add_edge(8, i)

    return G

# --- Simulate network of Chialvo neurons ---
def simulate_chialvo_network(G, steps=500, alpha=3.7, beta=0.01, coupling=0.005):
    """
    Simulates Chialvo neurons on the graph with optional diffusive coupling.
    x_i(t+1) = f(x_i) + coupling * sum_j( A_ij (x_j - x_i) )
    
    States are clipped into [0,1] to avoid numerical blow-up.
    """
    n = len(G.nodes())
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)

    # Initial random states inside (0,1)
    x = np.random.uniform(0.1, 0.9, size=n)
    states = np.zeros((steps, n))
    states[0] = x

    for t in range(1, steps):
        new_x = np.zeros(n)
        for i in range(n):
            # Chialvo update
            xi_next = chialvo_1d_step(x[i], alpha, beta)
            # Coupling term (diffusive)
            coupling_term = coupling * np.sum(A[i] * (x - x[i]))
            new_x[i] = xi_next + coupling_term

        # Keep states bounded to prevent blow-up
        new_x = np.clip(new_x, 0, 1)
        x = new_x
        states[t] = x

    return states, nodes

# --- Main ---
if __name__ == '__main__':
    # Build double-star network
    G = create_double_star_network()

    # Simulate network of Chialvo neurons
    steps = 500
    states, nodes = simulate_chialvo_network(G, steps=steps, alpha=3.7, beta=0.01, coupling=0.005)

    # Plot trajectories of all nodes
    plt.figure(figsize=(12, 6))
    for i, node in enumerate(nodes):
        plt.plot(states[:, i], lw=0.8, label=f"Node {node}")
    plt.title("Double-Star Network of Chialvo Neurons", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("x (neuron state)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=4, fontsize=8)
    plt.show()

    # Draw the network structure
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=800,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray"
    )
    plt.title("Double-Star Network Structure", fontsize=14)
    plt.show()
