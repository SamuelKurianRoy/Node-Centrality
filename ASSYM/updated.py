# -*- coding: utf-8 -*-
"""
Corrected script to reproduce the node centrality experiment from the paper
"Node Centrality Based on Edge Dynamics in a Chaotic Network".

Corrections made:
1.  Fixed the sign in the adaptive weight update law within the ODE function.
2.  Corrected the GEXF export logic to save the proper network files.
"""
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# ---------- Networks ----------
def create_double_star_13():
    """Creates the symmetric 13-node double-star network from the paper."""
    G = nx.Graph()
    G.add_nodes_from(range(1, 14))
    G.add_edge(1, 2)
    for i in range(3, 8): G.add_edge(2, i)
    G.add_edge(1, 8)
    for i in range(9, 14): G.add_edge(8, i)
    return G

def create_double_star_15():
    """Creates an asymmetric 15-node double-star network for comparison."""
    G = nx.Graph()
    G.add_nodes_from(range(1, 16))
    G.add_edge(1, 2)
    for i in range(3, 8): G.add_edge(2, i)
    G.add_edge(1, 8)
    for i in range(9, 16): G.add_edge(8, i)
    return G

# ---------- Centrality calculations ----------
def calculate_single_node_index(G):
    """Calculates the adjacent edge index (R) for each node."""
    nodelist = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]  # Eigenvector for 2nd smallest eigenvalue
    indices = {}
    for i, node in enumerate(nodelist):
        r_i = 0
        xi = fiedler_vector[i]
        for neighbor in G.neighbors(node):
            neighbor_idx = nodelist.index(neighbor)
            xj = fiedler_vector[neighbor_idx]
            r_i += abs(xi - xj)
        indices[node] = r_i
    return indices, fiedler_vector

def calculate_node_pair_matrix(G, single_node_indices, fiedler_vector):
    """Calculates the adjacent edge index for all node pairs (R^2)."""
    nodelist = sorted(G.nodes())
    num_nodes = len(nodelist)
    R2_matrix = np.zeros((num_nodes, num_nodes))
    for j_idx, node_j in enumerate(nodelist):
        for k_idx, node_k in enumerate(nodelist):
            if j_idx >= k_idx: continue
            R_j = single_node_indices[node_j]
            R_k = single_node_indices[node_k]
            # Subtract shared term if nodes are connected to avoid double-counting
            correction_term = 0
            if G.has_edge(node_j, node_k):
                x_j = fiedler_vector[j_idx]
                x_k = fiedler_vector[k_idx]
                correction_term = abs(x_j - x_k)
            
            R_jk = R_j + R_k - correction_term
            R2_matrix[j_idx, k_idx] = R_jk
    return R2_matrix

# ---------- Chua dynamics ----------
def chua_f(x):
    """Defines the dynamics of a single Chua's circuit."""
    p, q, r = x
    gamma1, gamma2, gamma3, gamma4 = 10.0, 18.0, -4.0 / 3.0, -3.0 / 4.0
    g_p = gamma4 * p + 0.5 * (gamma3 - gamma4) * (abs(p + 1) - abs(p - 1))
    return np.array([-gamma1 * (p - q + g_p), p - q + r, -gamma2 * q])

def make_coupled_chua_with_adaptive_edges(G, control_node_set, c=2.0, alpha=1.0):
    """Creates the ODE function for the coupled system with adaptive edge control."""
    nodes = sorted(G.nodes())
    N = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = sorted([(min(u, v), max(u, v)) for u, v in G.edges()])
    M = len(edges)

    # Identify edges connected to control nodes
    adaptive_mask = np.zeros(M)
    for ei, (u, v) in enumerate(edges):
        if (u in control_node_set) or (v in control_node_set):
            adaptive_mask[ei] = 1

    H = np.eye(3) # Inner-coupling matrix

    def ode_function(t, y):
        x_flat = y[:N * 3]
        w = y[N * 3:]
        X = x_flat.reshape((N, 3))

        # Build time-varying Laplacian matrix L(t) from edge weights
        A = np.zeros((N, N))
        for k, (u, v) in enumerate(edges):
            i, j = node_to_idx[u], node_to_idx[v]
            A[i, j] = w[k]
            A[j, i] = w[k]
        D = np.diag(A.sum(axis=1))
        L = D - A

        # Calculate node state derivatives (dx/dt)
        dx_list = [chua_f(X[i]) for i in range(N)]
        coupling_term = -c * (L @ X)
        dx = np.array(dx_list) + coupling_term

        # Calculate edge weight derivatives (dw/dt) for controlled edges
        dw = np.zeros(M)
        for k, (u, v) in enumerate(edges):
            if adaptive_mask[k]:
                i, j = node_to_idx[u], node_to_idx[v]
                diff = X[i] - X[j]
                # --- THIS IS THE CRITICAL CORRECTION ---
                # The sign must be negative as per the paper's formula.
                dw[k] = -alpha * float(diff @ (H @ diff))
        
        return np.concatenate([dx.flatten(), dw])

    return ode_function

# ---------- Simulation and Analysis ----------
def run_scenario(G, control_nodes, t_span=(0, 12), t_eval=None, seed=42):
    """Runs a single synchronization simulation."""
    if t_eval is None: t_eval = np.linspace(t_span[0], t_span[1], 600)
    
    ode_fun = make_coupled_chua_with_adaptive_edges(G, control_nodes)
    N, M = G.number_of_nodes(), G.number_of_edges()
    
    # Set initial conditions
    rng = np.random.RandomState(seed)
    x0 = (rng.rand(N * 3) - 0.5) * 10 # Random states from [-5, 5]
    w0 = np.ones(M) # Initial weights are 1
    y0 = np.concatenate([x0, w0])
    
    # Solve the ODE system
    sol = solve_ivp(fun=ode_fun, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-6, atol=1e-9, method='RK45')
    return sol

def sync_errors_from_solution(sol, N):
    """Calculates the synchronization error over time from a solution object."""
    x_traj = sol.y[:N * 3, :].reshape((N, 3, -1))
    T = x_traj.shape[2]
    errors = np.zeros((N, T))
    for t in range(T):
        avg_state = x_traj[:, :, t].mean(axis=0)
        for i in range(N):
            errors[i, t] = np.linalg.norm(x_traj[i, :, t] - avg_state)
    return errors

def time_to_sync(errors, t_eval, tol=1e-3):
    """Determines the time it takes for the network to synchronize."""
    max_err_over_time = errors.max(axis=0)
    sync_indices = np.where(max_err_over_time < tol)[0]
    return t_eval[sync_indices[0]] if len(sync_indices) > 0 else np.inf

# ---------- GEXF Export for Visualization (e.g., in Gephi) ----------
def export_enhanced_gexf(network_name, G_original):
    """Exports graph with centrality attributes to a GEXF file for visualization."""
    G = G_original.copy() # Work on a copy to avoid modifying the original
    
    single_indices, fiedler = calculate_single_node_index(G)
    
    # Add node attributes
    for i, node in enumerate(sorted(G.nodes())):
        G.nodes[node]['R_value'] = float(single_indices[node])
        G.nodes[node]['fiedler_component'] = float(fiedler[i])

    # Export to GEXF
    filename = f"{network_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.gexf"
    nx.write_gexf(G, filename)
    print(f"Enhanced GEXF exported to: {filename}")
    print(f"  - Node attributes: {list(next(iter(G.nodes(data=True)))[1].keys())}")

# ---------- Main Execution Block ----------
if __name__ == "__main__":
    networks = {
        "Paper Symmetric (13 nodes)": create_double_star_13(),
        "Asymmetric (15 nodes)": create_double_star_15()
    }
    
    results = []
    
    for name, G in networks.items():
        print(f"\n--- Analyzing: {name} ---")
        
        # 1. Calculate centrality indices
        single_indices, fiedler = calculate_single_node_index(G)
        R2_matrix = calculate_node_pair_matrix(G, single_indices, fiedler)
        nodelist = sorted(G.nodes())

        # 2. Find most important single node and pair from indices
        most_imp_node, max_r_node = max(single_indices.items(), key=lambda item: item[1])
        pair_idx = np.unravel_index(np.argmax(R2_matrix), R2_matrix.shape)
        most_imp_pair = (nodelist[pair_idx[0]], nodelist[pair_idx[1]])
        max_r_pair = R2_matrix[pair_idx]
        
        print(f"Most important node: {most_imp_node} (R = {max_r_node:.4f})")
        print(f"Most important pair: {most_imp_pair} (R² = {max_r_pair:.4f})")

        # 3. Run simulations to validate predictions
        print("Running simulations to find synchronization times...")
        t_eval = np.linspace(0, 12, 600)
        
        # Simulation for the single most important node
        sol_node = run_scenario(G, [most_imp_node], t_eval=t_eval)
        err_node = sync_errors_from_solution(sol_node, G.number_of_nodes())
        t_sync_node = time_to_sync(err_node, t_eval)

        # Simulation for the most important pair
        sol_pair = run_scenario(G, list(most_imp_pair), t_eval=t_eval)
        err_pair = sync_errors_from_solution(sol_pair, G.number_of_nodes())
        t_sync_pair = time_to_sync(err_pair, t_eval)

        print(f"  -> Sync time (controlling node {most_imp_node}): {t_sync_node:.2f} seconds")
        print(f"  -> Sync time (controlling pair {most_imp_pair}): {t_sync_pair:.2f} seconds")
        
        results.append([name, most_imp_node, max_r_node, t_sync_node, most_imp_pair, max_r_pair, t_sync_pair])

    # 4. Display summary table
    df = pd.DataFrame(results, columns=["Network", "Most Imp. Node", "R (Node)", "Sync Time (Node)",
                                        "Most Imp. Pair", "R² (Pair)", "Sync Time (Pair)"])
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    # --- THIS IS THE CORRECTED EXPORT LOGIC ---
    # 5. Export GEXF files for visualization
    print("\n" + "="*80)
    print("EXPORTING NETWORK FILES FOR VISUALIZATION")
    print("="*80)
    g13 = create_double_star_13()
    export_enhanced_gexf("Paper Symmetric (13 nodes)", g13)
    
    g15 = create_double_star_15()
    export_enhanced_gexf("Asymmetric (15 nodes)", g15)