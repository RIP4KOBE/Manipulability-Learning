import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import roboticstoolbox as rtb


def plot_gmm(Mu, Sigma, color, valAlpha=1, linestyle='-', linewidth=0.5, edgeAlpha=None):
    """
    This function displays the parameters of a Gaussian Mixture Model (GMM).

    Args:
    - Mu:         2xK array representing the centers of K Gaussians.
    - Sigma:      2x2xK array representing the covariance matrices of K Gaussians.
    - color:      Array representing the RGB color to use for the display.
    - valAlpha:   Transparency factor (optional, default=1).
    - linestyle:  Line style for ellipse edges (optional, default='-').
    - linewidth:  Line width for ellipse edges (optional, default=0.5).
    - edgeAlpha:  Transparency factor for edges (optional, default=same as valAlpha).

    Returns:
    - h: List of plot handles for the generated patches and points.
    - X: 2xN array of the ellipse points in each Gaussian.
    """

    nbStates = Mu.shape[1]
    nbDrawingSeg = 100
    darkcolor = np.clip(color * 0.5 * 2, 0, 1)  # Darker color for edges
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)

    if edgeAlpha is None:
        edgeAlpha = valAlpha

    h = []
    X = np.zeros((2, nbDrawingSeg, nbStates))

    fig, ax = plt.gca(), plt.gca()

    for i in range(nbStates):
        # Eigenvalue decomposition (for covariance matrices)
        D, V = np.linalg.eig(Sigma[:, :, i])
        R = np.real(V @ np.diag(np.sqrt(D)))  # Transformation matrix for ellipse

        X[:, :, i] = R @ np.array([np.cos(t), np.sin(t)]) + Mu[:, i, np.newaxis]

        # Plot with transparency
        polygon = Polygon(X[:, :, i].T, closed=True, facecolor=color, edgecolor=darkcolor,
                          alpha=valAlpha, linestyle=linestyle, linewidth=linewidth)
        ax.add_patch(polygon)
        h.append(polygon)

        # Plot the mean point
        point, = ax.plot(Mu[0, i], Mu[1, i], '.', color=darkcolor, markersize=6)
        h.append(point)

    ax.set_aspect('equal')
    plt.draw()

    return h, X


def plot_gmm_3d(Mu, Sigma, color, alpha=1, dispOpt=1):
    """
    This function displays the parameters of a Gaussian Mixture Model (GMM) in 3D.

    Args:
    - Mu:      3xK array representing the centers of K Gaussians.
    - Sigma:   3x3xK array representing the covariance matrices of K Gaussians.
    - color:   Array representing the RGB color to use for the display.
    - alpha:   Transparency factor (optional, default=1).
    - dispOpt: Display option for the plot (optional, default=1).

    Returns:
    - h: List of plot handles for the generated patches and points.
    """

    nbData = Mu.shape[1]
    nbPoints = 20  # Number of points to form a circular path
    nbRings = 10  # Number of circular paths following the principal direction

    pts0 = np.array([np.cos(np.linspace(0, 2 * np.pi, nbPoints)),
                     np.sin(np.linspace(0, 2 * np.pi, nbPoints))])

    h = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for n in range(nbData):
        # Eigenvalue decomposition (for covariance matrices)
        D0, V0 = np.linalg.eigh(Sigma[:, :, n])
        U0 = np.real(V0 @ np.diag(np.sqrt(D0)))

        # Generate the circular paths (rings)
        ringpts0 = np.array([np.cos(np.linspace(0, np.pi, nbRings + 1)),
                             np.sin(np.linspace(0, np.pi, nbRings + 1))])
        ringpts = np.zeros((3, nbRings))
        ringpts[1:, :] = ringpts0[:, :-1]
        U = np.zeros((3, 3))
        U[:, 1:] = U0[:, 1:]
        ringTmp = U @ ringpts

        # Compute the circular paths
        xring = np.zeros((3, nbPoints, nbRings))
        for j in range(nbRings):
            U = np.zeros((3, 3))
            U[:, 0] = U0[:, 0]
            U[:, 1] = ringTmp[:, j]
            pts = np.zeros((3, nbPoints))
            pts[:2, :] = pts0
            xring[:, :, j] = U @ pts + Mu[:, n, np.newaxis]

        # Close the ellipsoid
        xringfull = np.concatenate([xring, xring[:, :, [0]]], axis=2)

        # Plot the ellipsoid
        for j in range(nbRings):
            for i in range(nbPoints - 1):
                verts = [
                    [xringfull[:, i, j], xringfull[:, i + 1, j], xringfull[:, i + 1, j + 1], xringfull[:, i, j + 1]]
                ]
                poly = Poly3DCollection(verts, facecolor=np.minimum(color + 0.1, 1), edgecolor=color, alpha=alpha)
                if dispOpt == 1:
                    poly.set_edgecolor(color)
                    poly.set_linewidth(1)
                    poly.set_alpha(alpha)
                elif dispOpt == 2:
                    poly.set_linestyle('none')
                h.append(poly)
                ax.add_collection3d(poly)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    return h


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
    # 将 xax, yax, zax 转换为 3D 点的列表
    surface_points = np.vstack((xax.flatten(), yax.flatten(), zax.flatten()))

    # 旋转圆锥表面的所有点
    rotated_surface = rotate_points(surface_points, direction, angle_deg)

    # 重新调整形状到原始的网格形状
    X_rot = rotated_surface[0, :].reshape(xax.shape)
    Y_rot = rotated_surface[1, :].reshape(yax.shape)
    Z_rot = rotated_surface[2, :].reshape(zax.shape)

    # 绘制旋转后的圆锥表面
    ax.plot_surface(X_rot, Y_rot, Z_rot, color=[0.95, 0.95, 0.95], alpha=0.5)

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

def rtb_ik():
    np.set_printoptions(precision=6, suppress=True)
    robot = rtb.models.Panda()

    qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
    qz = np.zeros(7)

    print("正解")
    te = robot.fkine(qr)
    print(te.data[0])

    print("逆解")
    # 可能存在多个逆解，若不设置seed, 多次执行返回的结果可能不一样
    # q1 = robot.ikine_LM(te.data[0], q0=qz).q
    q1 = robot.ikine_LM(te.data[0], q0=qz, seed=1234).q
    print(q1)

    # 检查逆解是否正确
    assert np.allclose(te.data[0], robot.fkine(q1).data[0])
    print("逆解正确")

    # robot.plot(qr, backend="swift", block=True)


# main function
if __name__ == '__main__':

    # # plot_gmm() testing
    # Mu = np.array([[0, 2], [0, 3]])  # Mean of 2 Gaussians in 2D
    # Sigma = np.zeros((2, 2, 2))
    # Sigma[:, :, 0] = np.array([[1, 0.5], [0.5, 1]])
    # Sigma[:, :, 1] = np.array([[0.5, -0.2], [-0.2, 0.5]])
    # color = np.array([0.1, 0.2, 0.5])
    #
    # plot_gmm(Mu, Sigma, color, valAlpha=0.6)
    # plt.show()

    # #plot_gmm_3d testing
    # Mu = np.array([[0, 2, 1], [0, 3, 1], [0, 0, 1]]).T  # Mean of 3 Gaussians in 3D
    # Sigma = np.zeros((3, 3, 3))  # Adjust Sigma to have 3 covariance matrices
    # Sigma[:, :, 0] = np.array([[1, 0.3, 0.2], [0.3, 1, 0.1], [0.2, 0.1, 1]])
    # Sigma[:, :, 1] = np.array([[0.5, -0.1, 0], [-0.1, 0.5, 0], [0, 0, 0.5]])
    # Sigma[:, :, 2] = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, -0.2], [0.1, -0.2, 0.8]])  # Third Gaussian
    # color = np.array([0.1, 0.2, 0.5])
    #
    # plot_gmm_3d(Mu, Sigma, color, alpha=0.6)


    # plot_cone() testing
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 创建圆锥数据
    # r = 200
    # phi = np.linspace(0, 2 * np.pi, 100)
    # xax = np.array([np.zeros_like(phi), r * np.ones_like(phi)])
    # yax = np.array([np.zeros_like(phi), r * np.sin(phi)])
    # zax = np.array([np.zeros_like(phi), r / np.sqrt(2) * np.cos(phi)])
    #
    # # 定义旋转方向和角度
    # direction = np.cross([1, 0, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    # angle_deg = 45
    #
    # # 调用绘图函数
    # plot_cone(ax, xax, yax, zax, direction, angle_deg)
    #
    # # 显示图形
    # plt.show()

    # rtb_ik() testing
    rtb_ik()