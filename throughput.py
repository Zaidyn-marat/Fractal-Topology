import numpy as np
import networkx as nx
from itertools import combinations
import fractal_recursion
import matplotlib.pyplot as plt

dimension = 7
num_points = 100

x,y = fractal_recursion.generate_fractal_points(dimension, num_points)


coordinates = [(x[i], y[i]) for i in range(len(x))]

def create_mesh_network_with_coordinates(coordinates, max_distance):
    G = nx.Graph()  


    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=coord)  

    for i, j in combinations(range(len(coordinates)), 2): 
        distance = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
        if distance <= max_distance:
            G.add_edge(i, j, capacity=1 / (1 + distance),flow=100) 

    return G

def calculate_throughput(G, source, target):
    """
    Рассчет максимального потока в сети от source до target.
    """
    flow_value, _ = nx.maximum_flow(G, source, target, capacity='capacity')
    
    return flow_value

# def calculate_average_throughput(G):

#     nodes = list(G.nodes) 
#     total_throughput = 0
#     count = 0

#     for source, target in combinations(nodes, 2): 
#         try:
#             throughput = calculate_throughput(G, source, target)
#             total_throughput += throughput
#             count += 1
#         except nx.NetworkXError:
#             pass


#     average_throughput = total_throughput / count if count > 0 else 0
#     return average_throughput

max_distance = 0.9
plt.figure()
G = create_mesh_network_with_coordinates(coordinates, max_distance)
nx.draw(G, pos=coordinates)
# average_throughput = calculate_average_throughput(G)
# print(f"Средня пропускная способность беспроводной сети: {average_throughput:.2f}")
