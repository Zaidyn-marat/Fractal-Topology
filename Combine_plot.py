import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mplcyberpunk
from itertools import combinations
from matplotlib.colors import Normalize
from adjustText import adjust_text
from fractal_recursion import generate_fractal_points
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

# 🌟 Network Plot 相关函数
def get_node_colors(graph, cmap='coolwarm'):
    degrees = np.array([d for _, d in graph.degree()])
    norm_degrees = (degrees - degrees.min()) / (degrees.max() - 1e-5)
    return plt.get_cmap(cmap)(norm_degrees)


def create_mesh_network_with_coordinates(coordinates, max_distance):
    G = nx.Graph()
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=coord)
    for i, j in combinations(range(len(coordinates)), 2):
        distance = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
        if distance <= max_distance:
            G.add_edge(i, j)
    return G

def visualize_fractal_network(G, ax, cmap='coolwarm'):
    
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = get_node_colors(G, cmap)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, edgecolors='black', linewidths=0.8, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#005f99', width=0.5, alpha=0.7, ax=ax)
    mplcyberpunk.make_lines_glow(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"D = {D}", color='black', pad=5)


# 🌟 画布 & 布局
fig, axes_network = plt.subplots(3, 3, figsize=(12, 10), facecolor='white')
axes_network = axes_network.flatten()

# 生成并绘制 Network Plots (D=2 到 D=10)
for i, D in enumerate(range(1, 10)):
    num_points = 100
    max_distance = 0.2
    x, y = generate_fractal_points(D, num_points)
    coordinates = list(zip(x, y))
    G = create_mesh_network_with_coordinates(coordinates, max_distance)
    visualize_fractal_network(G, axes_network[i])

plt.tight_layout()
plt.savefig("Network.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()



# 🌟 单独绘制折线图
# 🌟 绘制双轴折线图
# D_values = np.arange(1, 11)

# # 性能指标数据
# metrics = {
#     "Throughput": [34.59, 33.75, 40.12, 39.66, 41.50, 41.91, 41.66, 39.74, 41.01, 38.67],
#     "Time Delay": [1.97, 1.10, 0.89, 0.96, 0.93, 0.94, 0.91, 0.91, 0.93, 1.04],
#     "Jitter": [0.85, 0.49, 0.35, 0.33, 0.43, 0.34, 0.36, 0.35, 0.48, 0.47],
#     "PDR": [0.39, 0.41, 0.48, 0.47, 0.48, 0.49, 0.49, 0.47, 0.48, 0.46],
# }

#---------------------------------------------------------------------------------------------------------------------------


# 数据准备
# D_values = np.arange(2, 11)
# metrics = {
#     "Throughput": [33.75, 40.12, 39.66, 41.50, 41.91, 41.66, 39.74, 41.01, 38.67],
#     "Time Delay": [1.10, 0.89, 0.96, 0.93, 0.94, 0.91, 0.91, 0.93, 1.04],
#     "Jitter": [0.49, 0.35, 0.33, 0.43, 0.34, 0.36, 0.35, 0.48, 0.47],
#     "PDR": [0.41, 0.48, 0.47, 0.48, 0.49, 0.49, 0.47, 0.48, 0.46],
# }
# norm = Normalize(vmin=-2, vmax=2)
# # 创建图形（白色背景）
# plt.style.use("default")  # 使用默认样式
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

# # --- 图1: Throughput vs PDR (保持不变) ---
# sc1 = ax1.scatter(
#     metrics["Throughput"], 
#     metrics["PDR"], 
#     c=D_values, 
#     cmap='coolwarm',
#     s=150, 
#     edgecolors='black',
#     linewidths=1,
#     alpha=0.9
# )

# # 标签位置调整
# label_positions = [
#     (0, 0), (0, 0), (0, 0.005), 
#     (0, 0), (0, 0.005), (0, 0.005),
#     (0, 0), (0, 0.005), (0, 0)
# ]

# for i, (x, y) in enumerate(zip(metrics["Throughput"], metrics["PDR"])):
#     dx, dy = label_positions[i]
#     ax1.text(
#         x + dx, y + dy + 0.005, 
#         f'D={D_values[i]}', 
#         ha='center', 
#         va='bottom', 
#         fontsize=8, 
#         color='black'
#     )

# # 添加趋势线
# x_trend = np.array(metrics["Throughput"])
# y_trend = np.array(metrics["PDR"])
# z = np.polyfit(x_trend, y_trend, 1)
# ax1.plot(x_trend, np.poly1d(z)(x_trend), '--', color='#1f77b4', lw=1.5)

# ax1.set_xlabel("Throughput (Mbps)", color='black')
# ax1.set_ylabel("Packet Delivery Ratio", color='black')
# # ax1.set_title("Throughput vs PDR", color='black')

# # --- 图2: Jitter vs Time Delay (坐标轴交换) ---
# sc2 = ax2.scatter(
#     metrics["Time Delay"],  # X轴改为Jitter
#     metrics["Jitter"],  # Y轴改为Time Delay
#     c=D_values, 
#     cmap='coolwarm', 
#     s=150, 
#     edgecolors='black',
#     linewidths=1,
#     alpha=0.9
# )

# # 标签位置调整
# label_positions = [
#     (0, 0.01), (0, 0), (0, 0), 
#     (0, 0), (0, 0), (0, 0),
#     (0, 0), (0, 0), (0, 0.01)
# ]

# for i, (x, y) in enumerate(zip(metrics["Time Delay"], metrics["Jitter"])):
#     dx, dy = label_positions[i]
#     ax2.text(
#         x + dx, y + dy + 0.008, 
#         f'D={D_values[i]}', 
#         ha='center', 
#         va='bottom', 
#         fontsize=8, 
#         color='black'
#     )
    

# ax2.set_xlabel("Time Delay (ms)", color='black')  # X轴标签
# ax2.set_ylabel("Jitter (ms)", color='black')  # Y轴标签
# # ax2.set_title("Jitter vs Time Delay", color='black')  # 标题更新

# # --- 通用设置 ---
# for ax in (ax1, ax2):
#     ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='gray')
#     ax.spines[['top', 'right']].set_visible(False)
    
#     # 移除cyberpunk效果，改用经典样式
#     for spine in ax.spines.values():
#         spine.set_color('black')
#     ax.tick_params(colors='black')

# # 添加颜色条
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(sc2, cax=cbar_ax)
# cbar.set_label('Fractal Dimension', rotation=270, labelpad=15, color='black')
# cbar.ax.yaxis.set_tick_params(color='black')
# for label in cbar.ax.get_yticklabels():
#     label.set_color('black')

# # 添加主标题
# # fig.suptitle("Fractal Network Performance Analysis", 
# #             fontsize=16, color='black', y=0.95)

# plt.tight_layout(rect=[0, 0, 0.9, 1])
# plt.savefig("fractal_performance_white.png", dpi=600, facecolor='white')
# plt.show()