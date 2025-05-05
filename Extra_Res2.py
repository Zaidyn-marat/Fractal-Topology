import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

def generate_fractal_points(dimension, num_points):

    points = []


    def generate_recursive(x, y, scale, depth):
        if depth == 0:
            return

        points.append((x, y))

        new_scale = scale / 4** (1 / dimension)
        generate_recursive(x - new_scale, y - new_scale, new_scale, depth - 1)
        generate_recursive(x + new_scale, y - new_scale, new_scale, depth - 1)
        generate_recursive(x - new_scale, y + new_scale, new_scale, depth - 1)
        generate_recursive(x + new_scale, y + new_scale, new_scale, depth - 1)
        
        # points.append((x, y))
        
    initial_scale = 0.5
    # max_depth =int(np.log2(num_points))
    max_depth = int(np.log((3 * num_points + 1) - 1) / np.log(4))
    # max_depth = int(np.log(3 * num_points + 1) / np.log(4)) - 1
    generate_recursive(0.5, 0.5, initial_scale, max_depth)


    points = np.array(points[:num_points])
    
    x = (points[:, 0] - min(points[:, 0])) / (max(points[:, 0]) - min(points[:, 0]))
    y = (points[:, 1] - min(points[:, 1])) / (max(points[:, 1]) - min(points[:, 1]))
    return x,y

# 创建网络（添加权重）
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

# 参数设置
num_points = 100
max_distance = 0.2  # 连接距离阈值
D_values = range(1, 11)

# 存储指标的字典
metrics = {
    'avg_degree': [],
    'avg_clustering': [],
    'avg_path_length': [],
    'diameter': [],
    'connectivity': []
}

# 主计算循环
for D in D_values:
    # 生成分形点
    x, y = generate_fractal_points(D, num_points)
    coordinates = list(zip(x, y))
    
    # 创建网络
    G = create_mesh_network_with_coordinates(coordinates, max_distance)
    
    # 计算基础指标
    degrees = [d for _, d in G.degree()]
    metrics['avg_degree'].append(np.mean(degrees))
    metrics['avg_clustering'].append(nx.average_clustering(G))
    
    # 处理连通分量
    if nx.is_connected(G):
        subG = G
        metrics['connectivity'].append(1.0)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()
        metrics['connectivity'].append(len(largest_cc)/num_points)
    
    # 路径相关指标（仅在连通分量节点数≥2时计算）
    if subG.number_of_nodes() >= 2:
        metrics['avg_path_length'].append(nx.average_shortest_path_length(subG))
        metrics['diameter'].append(nx.diameter(subG))
    else:
        metrics['avg_path_length'].append(0)
        metrics['diameter'].append(0)

# 可视化结果
plt.figure(figsize=(15, 10))

# 平均度
plt.subplot(2, 3, 1)
plt.plot(D_values, metrics['avg_degree'], 'bo-')
plt.xlabel('Fractal Dimension (D)')
plt.ylabel('Average Degree')
plt.title('Node Connectivity')

# 聚类系数
plt.subplot(2, 3, 2)
plt.plot(D_values, metrics['avg_clustering'], 'go-')
plt.xlabel('Fractal Dimension (D)')
plt.ylabel('Clustering Coefficient')
plt.title('Network Clustering')

# 平均路径长度
plt.subplot(2, 3, 3)
plt.plot(D_values, metrics['avg_path_length'], 'ro-')
plt.xlabel('Fractal Dimension (D)')
plt.ylabel('Average Path Length')
plt.title('Information Flow Efficiency')

# 网络直径
plt.subplot(2, 3, 4)
plt.plot(D_values, metrics['diameter'], 'co-')
plt.xlabel('Fractal Dimension (D)')
plt.ylabel('Network Diameter')
plt.title('Longest Shortest Path')

# 连通性
plt.subplot(2, 3, 5)
plt.plot(D_values, metrics['connectivity'], 'mo-')
plt.xlabel('Fractal Dimension (D)')
plt.ylabel('Connectivity Ratio')
plt.title('Network Robustness')

plt.tight_layout()
plt.show()