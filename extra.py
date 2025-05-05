import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import networkx as nx
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import pd


from itertools import combinations
from scipy.sparse.csgraph import shortest_path
from networkx.algorithms.flow import shortest_augmenting_path

# 生成 2D 分形点（保持不变）
def generate_fractal_points(dimension, num_points):
    points = []
    def generate_recursive(x, y, scale, depth):
        if depth == 0:
            return
        points.append((x, y))
        new_scale = scale / 4**(1 / dimension)
        generate_recursive(x - new_scale, y - new_scale, new_scale, depth - 1)
        generate_recursive(x + new_scale, y - new_scale, new_scale, depth - 1)
        generate_recursive(x - new_scale, y + new_scale, new_scale, depth - 1)
        generate_recursive(x + new_scale, y + new_scale, new_scale, depth - 1)
    initial_scale = 0.5
    max_depth = int(np.log((3 * num_points + 1) - 1) / np.log(4))
    generate_recursive(0.5, 0.5, initial_scale, max_depth)
    points = np.array(points[:num_points])
    return points[:, 0], points[:, 1]

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

# 计算网络中的加权最短路径距离矩阵
def compute_network_distance_matrix(G):
    length_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    num_nodes = len(G.nodes)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i, j] = length_dict[i].get(j, np.inf if i != j else 0)
    return dist_matrix

# 可视化函数（更新为新的高维数据）
def plot_linear_reduction(high_dim_data, original_2d):
    plt.figure(figsize=(15, 6))
    
    # 3D 网络嵌入空间
    ax1 = plt.subplot(131, projection='3d')
    ax1.scatter(high_dim_data[:, 0], high_dim_data[:, 1], high_dim_data[:, 2], 
                c='blue', alpha=0.6, depthshade=False)
    ax1.set_title('Network-Based 3D Embedding')
    ax1.grid(False)
    
    # PCA 投影到 2D
    pca = PCA(n_components=2)
    projected_pca = pca.fit_transform(high_dim_data)
    ax2 = plt.subplot(132)
    ax2.scatter(projected_pca[:, 0], projected_pca[:, 1], alpha=0.6,
                c='green', label='PCA Projection')
    ax2.set_title('PCA Projection with Network Features')
    ax2.axis('equal')
    
    # 原生 2D 分形
    ax3 = plt.subplot(133)
    ax3.scatter(original_2d[:, 0], original_2d[:, 1], alpha=0.6,
                c='red', label='Native 2D Fractal')
    ax3.set_title('Direct 2D Generation')
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.savefig('fractal_network_projection.png', dpi=300)
    plt.show()


D = 3
num_points =100
max_distance = 0.2

# 生成 2D 分形点
x, y = generate_fractal_points(D, num_points)
coordinates = list(zip(x, y))
original_2d = np.column_stack((x, y))

# 创建网络
G = create_mesh_network_with_coordinates(coordinates, max_distance)

# 计算加权最短路径距离矩阵
dist_matrix = compute_network_distance_matrix(G)

# 使用 MDS 嵌入到 3D
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
high_dim_data = mds.fit_transform(dist_matrix)

# 可视化
plot_linear_reduction(high_dim_data, original_2d)

# 可选：绘制网络图
plt.figure()
nx.draw(G, pos=coordinates, node_size=50, node_color='lightblue')
plt.title('Mesh Network')
plt.savefig('mesh_network.png', dpi=300)
plt.show()




# 分形生成验证函数
def validate_fractal_dim(points, scales=np.logspace(-2, 0, 10)):
    counts = []
    for ε in scales:
        count = 0
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        for x in np.arange(xmin, xmax, ε):
            for y in np.arange(ymin, ymax, ε):
                if np.any((points[:,0]>=x) & (points[:,0]<x+ε) & 
                         (points[:,1]>=y) & (points[:,1]<y+ε)):
                    count +=1
        counts.append(count)
    coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
    return coeffs[0]

# 网络性能分析函数
def analyze_network(G):
    # 平均最短路径
    avg_path = nx.average_shortest_path_length(G, weight='weight')
    
    # 网络容量（最大流求和）
    total_capacity = sum(data['capacity'] for _, _, data in G.edges(data=True))
    
    # 全局效率
    efficiency = nx.global_efficiency(G)
    
    # 路由复杂度
    betweenness = np.mean(list(nx.betweenness_centrality(G, weight='weight').values()))
    
    return {
        'avg_path': avg_path,
        'total_capacity': total_capacity,
        'efficiency': efficiency,
        'betweenness': betweenness
    }

# 维度优化主流程
def optimize_dimension(dims=np.arange(1, 11), num_points=300, max_distance=0.2):
    results = []
    
    for D in dims:
        # 生成分形点
        x, y = generate_fractal_points(D, num_points)
        points = np.column_stack((x, y))
        
        # 验证实际分形维度
        actual_dim = validate_fractal_dim(points)
        
        # 创建网络
        G = create_mesh_network_with_coordinates(list(zip(x, y)), max_distance)
        
        # 分析网络性能
        metrics = analyze_network(G)
        metrics.update({'dim_param': D, 'actual_dim': actual_dim})
        
        results.append(metrics)
    
    return pd.DataFrame(results)

# 运行优化
results_df = optimize_dimension()

# 可视化优化结果
plt.figure(figsize=(10,6))
plt.plot(results_df['dim_param'], results_df['efficiency'], 'bo-', label='Global Efficiency')
plt.plot(results_df['dim_param'], results_df['total_capacity'], 'rs--', label='Total Capacity')
plt.xlabel('Fractal Dimension Parameter')
plt.ylabel('Normalized Metric Value')
plt.title('Network Performance vs Fractal Dimension')
plt.legend()
plt.grid(True)
plt.savefig('dimension_optimization.png', dpi=300)
plt.show()