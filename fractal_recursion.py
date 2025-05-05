import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os 

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

dimensions = np.arange(1.5, 10, 1)
num_points = 86
saved_files = []

for d in dimensions:
    x, y = generate_fractal_points(d, num_points)
    coords = np.column_stack((x, y))
    filename = f"fractal_coordinates_dim_{round(d, 1)}.txt"
    np.savetxt(filename, coords, fmt="%.6f", delimiter="\t", header=f"Dimension: {d}", comments="")
    saved_files.append(filename)

saved_files


# import numpy as np
# import matplotlib.pyplot as plt

# def generate_fractal_points(dimension, num_points):
#     points = []
    
#     def generate_recursive(x, y, scale, depth):
#         if len(points) >= num_points:  # 只生成所需的点
#             return
#         points.append((x, y))
#         if depth == 0:
#             return
#         new_scale = scale / (4 ** (1 / dimension))
#         generate_recursive(x - new_scale, y - new_scale, new_scale, depth - 1)
#         generate_recursive(x + new_scale, y - new_scale, new_scale, depth - 1)
#         generate_recursive(x - new_scale, y + new_scale, new_scale, depth - 1)
#         generate_recursive(x + new_scale, y + new_scale, new_scale, depth - 1)

#     initial_scale = 0.5
#     max_depth = int(np.log(num_points) / np.log(4))  # 估算合适的递归深度

#     generate_recursive(0.5, 0.5, initial_scale, max_depth)
#     points = np.array(points[:num_points])  # 确保点数符合要求
#     return points[:, 0], points[:, 1]


# num_points_list = [1, 5, 21, 85, 341]

# # 创建子图
# fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=200, facecolor='black')

# for i, num_points in enumerate(num_points_list):
#     x, y = generate_fractal_points(dimension=1, num_points=num_points)
#     x = (x - min(x)) / (max(x) - min(x))
#     y = (y - min(y)) / (max(y) - min(y))

#     axes[i].set_facecolor('black')
#     axes[i].scatter(x, y, s=5, color='white', alpha=0.8)  # 适当放大点
#     axes[i].set_title(f"{num_points} Points", fontsize=12, color='white')
#     axes[i].set_xticks([])
#     axes[i].set_yticks([])
#     axes[i].spines[:].set_color('white')

# plt.tight_layout()
# plt.savefig("fractal_process.pdf", format="pdf", dpi=600, bbox_inches="tight", facecolor='black')
# plt.savefig("fractal_process.png", format="png", dpi=600, bbox_inches="tight", facecolor='black')




# dimension = 1
# num_points = 100
# x, y = generate_fractal_points(dimension, num_points)


# # 画图
# fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
# ax.set_facecolor('black')
# ax.scatter(x, y, s=1, color='white', alpha=0.8)

# # 设置坐标轴样式
# ax.tick_params(colors='white')  # 让刻度变白色
# ax.spines['bottom'].set_color('white')
# ax.spines['left'].set_color('white')
# ax.spines['top'].set_color('white')
# ax.spines['right'].set_color('white')


# fig.savefig("fractal_with_axes.pdf", format="pdf", dpi=600, bbox_inches="tight", facecolor='black')
 # fig.savefig("fractal_with_axes.png", format="png", dpi=600, bbox_inches="tight", facecolor='black')

# plt.show()






# num_points = 100
# for dimension in range(1, 11):
#     x, y = generate_fractal_points(dimension, num_points)
#     x = (x - min(x)) / (max(x) - min(x))
#     y = (y - min(y)) / (max(y) - min(y))
    
#     # file_path = os.path.join(output_folder, f"D{dimension}N{num_points}.txt")
#     # # np.savetxt(file_path, np.column_stack((x, y)), header="x y", comments='')
#     # print(f"Saved: {file_path}")
    
# plt.figure(figsize=(8, 8))
# plt.gca().set_facecolor('black')
# plt.scatter(x, y, s=1, color='white', alpha=0.8)
# plt.show()

# plt.savefig('trg.png', dpi=1000)




