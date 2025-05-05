import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from fractal_recursion import generate_fractal_points
# 参数设置
num_points = 100      # 节点数量
max_distance = 0.2   # 连接阈值（根据坐标范围0-1调整）
dim_range = range(1, 11)  # 分形维度范围

# 初始化存储结果
connectivity = []

def create_mesh_network_with_coordinates(coordinates, max_distance):
    G = nx.Graph()
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=coord)
    for i, j in combinations(range(len(coordinates)), 2):
        distance = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
        if distance <= max_distance:
            capacity = 1 / (1 + distance)
            G.add_edge(i, j, capacity=capacity, weight=1 / capacity)  # 权重为 1/容量
    return G

# 主计算循环
for D in dim_range:
    # 生成分形点
    x, y = generate_fractal_points(D, num_points)
    coordinates = list(zip(x, y))
    
    # 创建网络
    G = create_mesh_network_with_coordinates(coordinates, max_distance)
    
    # 计算连通度
    N = G.number_of_nodes()
    E = G.number_of_edges()
    possible_edges = N*(N-1)/2 if N > 1 else 0
    current_connectivity = E/possible_edges if possible_edges > 0 else 0
    connectivity.append(current_connectivity)

# 可视化结果
plt.figure(figsize=(12, 6))

# 连通度曲线
plt.subplot(1, 2, 1)
plt.plot(dim_range, connectivity, 'bo-', markersize=8)
plt.xlabel('Fractal Dimension (D)', fontsize=12)
plt.ylabel('Network Connectivity', fontsize=12)
plt.title('Connectivity vs Fractal Dimension', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(dim_range)

# 示例网络可视化（D=2和D=8）
plt.subplot(1, 2, 2)
for idx, example_D in enumerate([8], 1):
    x, y = generate_fractal_points(example_D, num_points)
    G = create_mesh_network_with_coordinates(list(zip(x, y)), max_distance)
    
    plt.scatter(x, y, s=30, label=f'D={example_D}')
    for edge in G.edges():
        plt.plot([x[edge[0]], x[edge[1]]], 
                 [y[edge[0]], y[edge[1]]], 
                 'gray', alpha=0.4, lw=0.8)
plt.title('Network Topology Comparison', fontsize=14)
plt.axis('equal')
plt.legend()
plt.tight_layout()

plt.show()