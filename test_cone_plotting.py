import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def rotate_points(points, axis, angle_deg):
    """
    旋转点集，类似于 MATLAB 的 rotate 函数。
    Parameters:
    - points: 点集 (3xN numpy array)
    - axis: 旋转轴 (3D numpy array)
    - angle_deg: 旋转角度（度）

    Returns:
    - 旋转后的点集
    """
    r = R.from_rotvec(np.radians(angle_deg) * axis)
    return r.apply(points.T).T


def plot_cone(ax, xax, yax, zax, direction, angle_deg):
    """
    在给定的 3D 轴上绘制操控性椭圆体的圆锥体。

    Parameters:
    - ax: matplotlib 的 3D 轴
    - xax, yax, zax: 定义圆锥的 X, Y, Z 数据
    - direction: 旋转方向向量
    - angle_deg: 旋转角度
    """
    # 绘制圆锥表面
    ax.plot_surface(xax, yax, zax, color=[0.95, 0.95, 0.95], alpha=0.5)

    # 绘制圆锥边缘线
    edge_line = np.vstack((xax[1, :], yax[1, :], zax[1, :]))
    rotated_edge = rotate_points(edge_line, direction, angle_deg)
    ax.plot(rotated_edge[0, :], rotated_edge[1, :], rotated_edge[2, :], color='black', linewidth=3)

    # 绘制生成线
    for idx in [63, 40]:
        generator_line = np.vstack((xax[:, idx], yax[:, idx], zax[:, idx]))
        rotated_gen_line = rotate_points(generator_line, direction, angle_deg)
        ax.plot(rotated_gen_line[0, :], rotated_gen_line[1, :], rotated_gen_line[2, :], color='black', linewidth=3)

    # 绘制坐标轴
    ax.plot([0, 250], [0, 0], [0, 0], 'k-', linewidth=0.5)
    ax.plot([0, 0], [0, 250], [0, 0], 'k-', linewidth=0.5)
    ax.plot([0, 0], [0, 0], [0, 150], 'k-', linewidth=0.5)

    # 添加文本标签
    ax.text(280, -40, 0, r'$\mathbf{M}_{11}$', fontsize=20, ha='center')
    ax.text(15, 0, 120, r'$\mathbf{M}_{12}$', fontsize=20, ha='center')
    ax.text(5, 220, -15, r'$\mathbf{M}_{22}$', fontsize=20, ha='center')

    # 设置图形的视角和隐藏刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()  # 隐藏轴
    ax.view_init(elev=12, azim=70)


# 测试函数
if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 创建圆锥数据
    r = 200
    phi = np.linspace(0, 2 * np.pi, 100)
    xax = np.array([np.zeros_like(phi), r * np.ones_like(phi)])
    yax = np.array([np.zeros_like(phi), r * np.sin(phi)])
    zax = np.array([np.zeros_like(phi), r / np.sqrt(2) * np.cos(phi)])

    # 定义旋转方向和角度
    direction = np.cross([1, 0, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    angle_deg = 0

    # 调用绘图函数
    plot_cone(ax, xax, yax, zax, direction, angle_deg)

    # 显示图形
    plt.show()