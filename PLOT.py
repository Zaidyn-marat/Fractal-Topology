import numpy as np
import networkx as nx
import fractal_recursion
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib import cm

# Generate fractal-based points
dimension = 9
num_points = 86
x, y = fractal_recursion.generate_fractal_points(dimension, num_points)
x = (x - min(x)) / (max(x) - min(x))
y = (y - min(y)) / (max(y) - min(y))

x=x/2
y=y/2 

coordinates = [(x[i], y[i]) for i in range(len(x))]

# Function to create the network
def create_mesh_network_with_coordinates(coordinates, max_distance):
    G = nx.Graph()
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=coord)

    for i, j in combinations(range(len(coordinates)), 2):
        distance = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
        if distance <= max_distance:
            G.add_edge(i, j)

    return G

# Function to plot the network beautifully
def plot_beautiful_network(G, coordinates):
    pos = {i: coord for i, coord in enumerate(coordinates)}

    plt.figure(figsize=(20, 20), dpi=300)

    # Custom aesthetic parameters
    edge_color = "#000000"  # Light blue for edges
    node_color = "#b60033"  # Bright pink for nodes

    # Draw edges
    for u, v in G.edges():
        n1, n2 = pos[u], pos[v]
        plt.plot(
            [n1[0], n2[0]], [n1[1], n2[1]],
            color=edge_color, alpha=1, linewidth=1
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=15,
        node_color=node_color, alpha=1
    )


    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Create and plot the network
max_distance = 0.15
G = create_mesh_network_with_coordinates(coordinates, max_distance)
plot_beautiful_network(G, coordinates)
