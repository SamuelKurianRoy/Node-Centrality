import matplotlib.animation as animation
from IPython.display import HTML
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Chialvo Neuron and Simulation Logic ---

def simulate_chaotic_neuron_network(G_initial, time_steps, chaos_strength=0.1, coupling_strength=0.05):
    """
    Simulates the evolution of a network where each node is a Chialvo neuron
    and the network structure is rewired chaotically.
    """
    # Chialvo neuron parameters
    a, b, c, k = 0.89, 0.6, 0.28, 0.01

    # Initialize neuron states (x, y) for each node
    nodes = list(G_initial.nodes())
    x_states = {node: np.random.rand() for node in nodes}
    y_states = {node: np.random.rand() for node in nodes}

    # Store the history of networks and neuron activity
    evolving_networks = [G_initial.copy()]
    activity_history = [{node: x_states[node] for node in nodes}]

    G_current = G_initial.copy()

    for t in range(time_steps - 1):
        # --- A: Structural Chaos (Stochastic Rewiring) ---
        G_new = G_current.copy()
        num_edges = len(G_new.edges())
        num_edges_to_change = min(int(num_edges * chaos_strength), num_edges)

        if num_edges_to_change > 0:
            # Randomly remove edges
            edges_to_remove = list(G_new.edges())
            indices = np.random.choice(range(len(edges_to_remove)), size=num_edges_to_change, replace=False)
            G_new.remove_edges_from([edges_to_remove[i] for i in indices])

            # Randomly add the same number of new edges
            for _ in range(num_edges_to_change):
                u, v = np.random.choice(nodes, size=2, replace=False)
                if not G_new.has_edge(u, v):
                    G_new.add_edge(u, v)
        
        evolving_networks.append(G_new)
        G_current = G_new

        # --- B: Nodal Chaos (Update Chialvo Neuron States) ---
        next_x_states = {}
        next_y_states = {}
        current_activity = {}

        for node in nodes:
            # Calculate input from connected neighbors
            neighbor_input = 0
            if G_current.has_node(node): # Ensure node still exists
                for neighbor in G_current.neighbors(node):
                    neighbor_input += x_states[neighbor]
            
            # Get previous states
            x_n, y_n = x_states[node], y_states[node]

            # Chialvo map equations
            x_n_plus_1 = (x_n**2) * np.exp(y_n - x_n) + k + (coupling_strength * neighbor_input)
            y_n_plus_1 = a * y_n - b * x_n + c

            next_x_states[node] = x_n_plus_1
            next_y_states[node] = y_n_plus_1
            current_activity[node] = x_n_plus_1
        
        x_states, y_states = next_x_states, next_y_states
        activity_history.append(current_activity)

    return evolving_networks, activity_history

# --- 2. Analysis and Visualization (Updated for Neuron Activity) ---

def extract_activity_trajectories(activity_history, num_nodes):
    """
    Formats the activity history for the visualization function.
    """
    trajectories = {node: {'time_points': [], 'activity_value': []} for node in range(num_nodes)}
    
    for t, snapshot in enumerate(activity_history):
        for node, activity in snapshot.items():
            trajectories[node]['time_points'].append(t)
            trajectories[node]['activity_value'].append(activity)
            
    return trajectories

def visualize_chaotic_evolution(evolving_networks, activity_trajectories, sample_rate=5):
    """Create an animated visualization of network and neuron evolution."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    sampled_networks = evolving_networks[::sample_rate]
    sampled_times = range(0, len(evolving_networks), sample_rate)

    def update(frame):
        for ax in axes.flat:
            ax.clear()

        G_current = sampled_networks[frame]
        current_time = sampled_times[frame]

        # 1. Network visualization with activity coloring
        pos = nx.spring_layout(G_current, seed=42)
        node_colors = [activity_trajectories[node]['activity_value'][current_time]
                       for node in G_current.nodes()]

        vmin = min(min(c) for c in (d['activity_value'] for d in activity_trajectories.values()))
        vmax = max(max(c) for c in (d['activity_value'] for d in activity_trajectories.values()))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # THIS IS THE CORRECTED LINE:
        nx.draw_networkx_nodes(G_current, pos, ax=axes[0,0], node_color=node_colors,
                               node_size=200, cmap='plasma', vmin=vmin, vmax=vmax)

        nx.draw_networkx_edges(G_current, pos, ax=axes[0,0], alpha=0.5, edge_color='gray')
        axes[0,0].set_title(f'Network Structure (Time: {current_time})')
        axes[0,0].set_axis_off()

        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[0,0], label='Neuron Activity (x-value)')

        # 2. Activity trajectory plot
        top_5_nodes = sorted(activity_trajectories.keys(),
                               key=lambda x: np.mean(activity_trajectories[x]['activity_value']),
                               reverse=True)[:5]

        for node in top_5_nodes:
            times = activity_trajectories[node]['time_points'][:current_time+1]
            values = activity_trajectories[node]['activity_value'][:current_time+1]
            axes[0,1].plot(times, values, label=f'Node {node}', linewidth=2)
            if values:
                axes[0,1].scatter([current_time], [values[-1]], s=50)

        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('Neuron Activity (x-value)')
        axes[0,1].set_title('Activity of Top 5 Nodes')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. Activity distribution histogram
        current_activities = [activity_trajectories[node]['activity_value'][current_time]
                               for node in G_current.nodes()]
        axes[1,0].hist(current_activities, bins=20, alpha=0.7, color='skyblue', range=(vmin, vmax))
        mean_activity = np.mean(current_activities)
        axes[1,0].axvline(mean_activity, color='red', linestyle='--', label=f'Mean: {mean_activity:.3f}')
        axes[1,0].set_xlabel('Neuron Activity (x-value)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Network Activity Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 4. Network metrics over time
        time_points = list(range(current_time + 1))
        densities = [nx.density(evolving_networks[t]) for t in time_points]
        avg_degrees = [np.mean([d for n, d in evolving_networks[t].degree()]) for t in time_points]

        axes[1,1].plot(time_points, densities, label='Network Density', linewidth=2)
        ax2 = axes[1,1].twinx()
        ax2.plot(time_points, avg_degrees, label='Average Degree', linewidth=2, color='orange')
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('Density', color='blue')
        ax2.set_ylabel('Avg. Degree', color='orange')
        axes[1,1].set_title('Network Metrics Over Time')
        axes[1,1].grid(True, alpha=0.3)
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.45))


        fig.suptitle(f'Chaotic Neuron Network Evolution - Time Step: {current_time}', fontsize=16)

    anim = animation.FuncAnimation(fig, update, frames=len(sampled_networks), interval=200)
    return HTML(anim.to_jshtml())

# --- 3. Run the Simulation and Display ---

# Initial setup
NUM_NODES = 50
TIME_STEPS = 200
G_initial = nx.erdos_renyi_graph(NUM_NODES, 0.15) # A slightly denser initial graph
pos = nx.spring_layout(G_initial, seed=42)

# Draw initial network
nx.draw(G_initial, pos, with_labels=True, node_size=1000, node_color="skyblue", edge_color="gray")
plt.title("Initial Network", fontsize=14)
plt.show()
# Run the full simulation
evolving_networks, activity_history = simulate_chaotic_neuron_network(
    G_initial, 
    time_steps=TIME_STEPS, 
    chaos_strength=0.2, # % of edges rewired each step
    coupling_strength=0.02 # How much neighbors influence a neuron
)

# Format data and visualize
activity_trajectories = extract_activity_trajectories(activity_history, NUM_NODES)
print("Generating animation... this might take a moment.")
animation_output = visualize_chaotic_evolution(evolving_networks, activity_trajectories)

# In a Jupyter Notebook, the following line would display the animation
# animation_output


# Save the animation to an HTML file
with open("network_animation.html", "w") as f:
    # f.write(animation_output.to_html5_video()) # Or anim.to_jshtml() for interactive JS
    # This is the correct line
    f.write(animation_output.data)

print("✅ Animation saved successfully to 'network_animation.html'")