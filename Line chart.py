import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns

# 🌟 数据
D_values = np.arange(2, 12)
metrics = {
    "Throughput": [34.59, 33.75, 40.12, 39.66, 41.50, 41.91, 41.66, 39.74, 41.01, 38.67],
    "PDR": [0.39, 0.41, 0.48, 0.47, 0.48, 0.49, 0.49, 0.47, 0.48, 0.46]
}

# 🎨 画布 & 背景
fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')  # 白色背景
ax.set_facecolor('white')  # 坐标区域背景

# 🎨 颜色映射 (低维=蓝色, 高维=红色)
cmap = sns.color_palette("coolwarm", as_cmap=True)
colors = [cmap((d - min(D_values)) / (max(D_values) - min(D_values))) for d in D_values]

# 🔵 颜色 & 形状映射维度信息
for i, (td, pdr, d) in enumerate(zip(metrics["Throughput"], metrics["PDR"], D_values)):
    shape = 'o' if d <= 5 else '^'  # 低维圆形, 高维三角形
    ax.scatter(td, pdr, color=colors[i], edgecolors='black', linewidth=0.8,
               s=150 + 10 * d, alpha=0.9, marker=shape)  # 维度越大, 点越大

    # 🏷️ 标注 D 值
    txt = ax.text(td, pdr, f"D={d}", fontsize=10, ha='right', va='bottom', color="black")
    txt.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground="white"),  # 白色描边增强可读性
        path_effects.Normal()
    ])

# 🎯 视觉优化
ax.set_xlabel("Throughput", fontsize=12, color='black')
ax.set_ylabel("PDR", fontsize=12, color='black')

# 📉 柔和网格线
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')

# 📊 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(D_values), vmax=max(D_values)))
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
cbar.set_label("Dimension (D)", fontsize=12, color='black')
cbar.ax.tick_params(labelsize=10, colors='black')

# 📁 展示图像
plt.show()
