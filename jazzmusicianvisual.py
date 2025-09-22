import os
import networkx as nx
import matplotlib.pyplot as plt
import re

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
    nx.write_gexf(G_edge,"jazz_musicians.gexf")

else:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure your .wl file is in the same directory as this notebook or provide the full path.")