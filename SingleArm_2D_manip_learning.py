import numpy as np
from scipy.interpolate import splev, splrep
from scipy.linalg import block_diag


import roboticstoolbox as rtb
from spatialmath import SE3
from manip_utils import *

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# Set random seed for reproducibility
np.random.seed(42)


# Now the complete main function
def ManipulabilityLearning():

    #<editor-fold desc="Parameter Initialization">

    nbData = 100  # Number of datapoints in a trajectory
    nbSamples = 4  # Number of demonstrations
    nbIter = 10  # Number of iteration for the Gauss Newton algorithm (Riemannian manifold)
    nbIterEM = 10  # Number of iteration for the EM algorithm
    letter = 'C'  # Letter to use as dataset for demonstration data

    # GMM model for Manipulability Ellipsoids
    modelPD = {
        'nbStates': 5,  # Number of Gaussians in the GMM over man. ellipsoids
        'nbVar': 3,  # Dimension of the manifold and tangent space (1D input (time t) + 2^2 output (symmetric SPD matrices of ME))
        'nbVarOut': 2,  # Dimension of the output (symmetric SPD matrices of ME)
        'dt': 1E-2,  # Time step duration
        'params_diagRegFact': 1E-4,  # Regularization of covariance
        'Kp': 100  # Gain for position control in task space
    }

    # Dimension of the output SPD covariance matrices
    modelPD['nbVarOutVec'] = modelPD['nbVarOut'] + modelPD['nbVarOut'] * (modelPD['nbVarOut'] - 1) // 2
    # Dimension of the manifold and tangent space in vector form
    modelPD['nbVarVec'] = modelPD['nbVar'] - modelPD['nbVarOut'] + modelPD['nbVarOutVec']
    # Dimension of the output SPD covariance matrices
    modelPD['nbVarCovOut'] = modelPD['nbVar'] + modelPD['nbVar'] * (modelPD['nbVar'] - 1) // 2

    # GMM model for Cartesian trajectories
    modelKin = {
        'nbStates': 5,  # Number of states in the GMM over 2D Cartesian trajectories
        'nbVar': 3,  # Number of variables [t,x1,x2]
        'dt': modelPD['dt']  # Time step duration
    }

    #</editor-fold>

    # <editor-fold desc="Create Robots">
    # Robots parameters
    nbDOFs = 3  # Number of degrees of freedom for teacher robot
    armLength = 5  # For letter 'C'

    # Define the robot using standard DH parameters
    L = [rtb.RevoluteDH(a=armLength, d=0, alpha=0) for _ in range(nbDOFs)]
    # Define the robot using Modified DH parameters
    # L = [rtb.RevoluteMDH(a=armLength, d=0, alpha=0) for _ in range(nbDOFs)]
    robotT = rtb.DHRobot(L, name='TeacherRobot')
    q0T = np.array([np.pi/4, 0.0, -np.pi/9])  # Initial robot configuration
    # </editor-fold>

    #<editor-fold desc="Load Demonstration Data and Generate Manipulability Ellipsoids">
    print('Loading demonstration data...')
    # Load the data
    import scipy.io
    data = scipy.io.loadmat(f'data/2Dletters/{letter}.mat')
    demos_cell = data['demos'][0]
    demos = []
    for demo in demos_cell:
        pos = demo['pos'][0,0]
        demos.append({'pos': pos})

    xIn = np.arange(1, nbData+1) * modelPD['dt']  # Time as input variable
    X = np.zeros((3, 3, nbData*nbSamples))  # Matrix storing t,x1,x2 and ME SPD Matrices (3x3) and data points (400) for all the demos
    X[0,0,:] = np.tile(xIn, nbSamples)
    # X[0, 0, :] = np.reshape(np.tile(xIn, nbSamples), (nbData * nbSamples))

    Data = []
    s = [{} for _ in range(nbSamples)]

    for n in range(nbSamples):
        s[n]['Data'] = []
        demo_pos = demos[n]['pos']
        # Spline interpolation
        tck_x = splrep(np.arange(demo_pos.shape[1]), demo_pos[0, :], s=0)
        tck_y = splrep(np.arange(demo_pos.shape[1]), demo_pos[1, :], s=0)
        dTmp_x = splev(np.linspace(0, demo_pos.shape[1]-1, nbData), tck_x)
        dTmp_y = splev(np.linspace(0, demo_pos.shape[1]-1, nbData), tck_y)
        s[n]['Data'] = np.vstack([dTmp_x, dTmp_y])

        # Obtain robot configurations for the current demo given initial robot pose q0
        # T_list = [SE3([s[n]['Data'][0, t], s[n]['Data'][1, t], 0]) for t in range(nbData)]
        # q = np.zeros((nbDOFs, nbData))
        # for t in range(nbData):
        #     sol = robotT.ikine_LM(Tep=T_list[t], q0=q0T, mask=[1,1,0,0,0,0])
        #     # solver = rtb.IK_LM(mask=[1, 1, 0, 0, 0, 0])
        #     # sol = solver.solve(robotT, Tep=T_list[t])
        #     q[:, t] = sol.q
        # s[n]['q'] = q  # Storing joint values

        # load demo q value for testing
        q_data = scipy.io.loadmat('data/test_joint_angle/s_data.mat')  # 读取 .mat 文件
        q_array = q_data['s']
        q = q_array[0, n]['q'] # 提取 `q` 字段
        s[n]['q'] = q

        # Computing force/velocity manipulability ellipsoids
        for t in range(nbData):
            J = robotT.jacob0(s[n]['q'][:, t], half='trans')[:2, :]  # Take only translational part
            # print("ME: ", J @ J.T)
            JJ_T = J @ J.T
            if is_spd(JJ_T):
                X[1:3, 1:3, t + n*nbData] = JJ_T  # Saving manipulability ellipsoid
                # print(f"{t + n*nbData}th manipulability ellipsoid is saved")
            else:
                raise ValueError(f"{t + n*nbData}th manipulability ellipsoid is not SPD")

        Data.append(np.vstack((xIn, s[n]['Data'])))

    Data = np.hstack(Data)  # Combining data from all samples
    # SPD data in vector shape
    x = np.zeros((4, nbData*nbSamples))
    x[0, :] = X[0, 0, :]
    for i in range(nbData*nbSamples):
        x[1:, i] = symmat2vec(X[1:, 1:, i])
    # </editor-fold>

    #<editor-fold desc="GMM Learning">
    print('Learning GMM1 (2D Cartesian position)...')
    modelKin_init = init_GMM_timeBased(Data, modelKin)
    # print("initial modelKin: ", modelKin)
    modelKin = EM_GMM(Data, modelKin_init)
    # print("learned modelKin: ", modelKin)

    print('Learning GMM2 (Manipulability ellipsoids)...')
    # Initialization on the manifold
    in_idx = 0
    outMat = np.arange(1, modelPD['nbVar'])
    out = np.arange(1, modelPD['nbVarVec'])
    modelPD = spd_init_GMM_kbins(x, modelPD, nbSamples, out)
    modelPD['Mu'] = np.zeros_like(modelPD['MuMan'])
    L = np.zeros((modelPD['nbStates'], nbData * nbSamples), dtype=np.float64)
    xts = np.zeros((modelPD['nbVarVec'], nbData * nbSamples, modelPD['nbStates']))

    # EM for SPD matrices manifold
    for nb in range(nbIterEM):
        print('.', end='')
        # E-step
        for i in range(modelPD['nbStates']):
            xts[in_idx, :, i] = x[in_idx, :] - modelPD['MuMan'][in_idx, i]
            xts[out, :, i] = logmap_vec(x[out, :], modelPD['MuMan'][out, i])
            L[i, :] = modelPD['Priors'][i] * gaussPDF(xts[:, :, i], modelPD['Mu'][:, i], modelPD['Sigma'][:, :, i])

        # Responsibilities
        L_sum = np.sum(L, axis=0, keepdims=True) + np.finfo(float).eps
        GAMMA = L / L_sum
        GAMMA_sum = np.sum(GAMMA, axis=1, keepdims=True) + np.finfo(float).eps
        H = GAMMA / GAMMA_sum

        # M-step
        for i in range(modelPD['nbStates']):
            # Update Priors
            modelPD['Priors'][i] = np.sum(GAMMA[i, :]) / (nbData * nbSamples)

            # Update MuMan
            for n in range(nbIter):
                # Update on the tangent space
                uTmp = np.zeros((modelPD['nbVarVec'], nbData * nbSamples))
                uTmp[in_idx, :] = x[in_idx, :] - modelPD['MuMan'][in_idx, i]
                uTmp[out, :] = logmap_vec(x[out, :], modelPD['MuMan'][out, i])
                uTmpTot = np.sum(uTmp * H[i, :], axis=1)

                # Update on the manifold
                modelPD['MuMan'][in_idx, i] = uTmpTot[in_idx] + modelPD['MuMan'][in_idx, i]
                modelPD['MuMan'][out, i] = expmap_vec(uTmpTot[out], modelPD['MuMan'][out, i])

            # Update Sigma
            modelPD['Sigma'][:, :, i] = (uTmp @ np.diag(H[i, :]) @ uTmp.T +
                                         np.eye(modelPD['nbVarVec']) * modelPD['params_diagRegFact'])

    # Eigendecomposition of Sigma
    V = np.zeros((modelPD['nbVarVec'], modelPD['nbVarVec'], modelPD['nbStates']))
    D = np.zeros((modelPD['nbVarVec'], modelPD['nbVarVec'], modelPD['nbStates']))

    for i in range(modelPD['nbStates']):
        D_matrices, V_matrices = np.linalg.eig(modelPD['Sigma'][:, :, i])
        V[:, :, i] = V_matrices
        D[:, :, i] = np.diag(D_matrices)

    print('GMM2 Learning Completed.')
    # </editor-fold>

    # <editor-fold desc="GMR Regression">
    # Initialization
    print('Regression...')
    xIn = np.zeros((1, nbData))
    xIn[0, :] = np.arange(1, nbData + 1) * modelPD['dt']

    in_idx = 0  # time index
    out = np.arange(1, modelPD['nbVarVec'])  # Output dimensions
    nbVarOut = len(out)
    outMan = np.arange(1, modelPD['nbVar'])

    # Initializations for GMR for manipulability ellipsoids
    uhat = np.zeros((nbVarOut, nbData))
    xhat = np.zeros((nbVarOut, nbData))
    uOut = np.zeros((nbVarOut, modelPD['nbStates'], nbData))
    expSigma = np.zeros((nbVarOut, nbVarOut, nbData))
    H = np.zeros((modelPD['nbStates'], nbData))
    xd = np.zeros((2, nbData))
    sigma_xd = np.zeros((2, 2, nbData))

    for t in range(nbData):
        # GMR for 2D Cartesian trajectory
        xd[:, t], sigma_xd[:, :, t], _ = GMR(modelKin, (t+1) * modelKin['dt'], in_idx, np.arange(1, modelKin['nbVar']))
        # print("2D Cartesian trajectory GMR finished!")

        # GMR for manipulability ellipsoids
        for i in range(modelPD['nbStates']):
            # Compute activation weight H(i,t): the activation weight of each state i,
            # indicating the probability that the data point belongs to the ith Gaussian component.
            H[i, t] = modelPD['Priors'][i] * gaussPDF(xIn[:, t] - modelPD['MuMan'][in_idx, i],
                                                      modelPD['Mu'][in_idx, i], modelPD['Sigma'][in_idx, in_idx, i]).item()
        H[:, t] /= np.sum(H[:, t]) + np.finfo(float).eps

        # Compute conditional mean (with covariance transportation)
        if t == 0:
            id_max = np.argmax(H[:, t])
            xhat[:, t] = modelPD['MuMan'][out, id_max]  # Initial point
        else:
            xhat[:, t] = xhat[:, t - 1]

        # Iterative computation
        for n in range(nbIter):
            uhat[:, t] = np.zeros(nbVarOut)
            for i in range(modelPD['nbStates']):
                # Transportation of covariance from model.MuMan(outMan,i) to xhat(:,t)
                S1 = vec2symmat(modelPD['MuMan'][out, i])
                S2 = vec2symmat(xhat[:, t])
                Ac = block_diag(1, transp_operator(S1, S2))  #TODO: ValueError: S2 is not a valid SPD matrix

                # Parallel transport of eigenvectors
                vMat = np.zeros((modelPD['nbVarOutVec'], modelPD['nbVarOutVec'], V.shape[1], modelPD['nbStates']))
                pvMat = np.zeros_like(vMat)
                pV = np.zeros((modelPD['nbVarVec'], modelPD['nbVarVec'], modelPD['nbStates']))
                pSigma = np.zeros((modelPD['nbVarVec'], modelPD['nbVarVec'], modelPD['nbStates']))

                for j in range(V.shape[1]):
                    vMat[:, :, j, i] = block_diag(block_diag(V[in_idx, j, i]), vec2symmat(V[out, j, i]))
                    # pvMat[:, :, j, i] = Ac @ np.sqrt(D[j, j, i]) * vMat[:, :, j, i] @ Ac.T

                    if np.isscalar(D[j, j, i]):  # 检查 D[j, j, i] 是否为标量
                        D_sqrt = np.sqrt(D[j, j, i])
                        pvMat[:, :, j, i] = Ac @ (D_sqrt * vMat[:, :, j, i]) @ Ac.T  # 使用标量乘法
                    else:
                        D_sqrt = np.sqrt(D[j, j, i])
                        pvMat[:, :, j, i] = Ac @ D_sqrt @ vMat[:, :, j, i] @ Ac.T  # 使用矩阵乘法


                    if np.isscalar(pvMat[in_idx, in_idx, j, i]):
                        pvMat_diag = np.array([pvMat[in_idx, in_idx, j, i]])  # 标量直接使用
                    else:
                        pvMat_diag = np.diag(pvMat[in_idx, in_idx, j, i])  # 如果是矩阵，提取对角线
                    pV[:, j, i] = np.concatenate(
                        [pvMat_diag, symmat2vec(pvMat[np.ix_(outMat, outMat, [j], [i])][:, :, 0, 0])])

                # Parallel transported sigma (reconstruction from eigenvectors)
                pSigma[:, :, i] = pV[:, :, i] @ pV[:, :, i].T
                # pSigma[:, :, i] = np.outer(pV[:, :, i], pV[:, :, i])


                # Gaussian conditioning on the tangent space
                if np.isscalar(pSigma[in_idx, in_idx, i]):
                    # 如果是标量，直接取倒数
                    inv_pSigma_in = 1 / pSigma[in_idx, in_idx, i]
                    uOut[:, i, t] = (logmap_vec(modelPD['MuMan'][out, i], xhat[:, t]).reshape(3, 1) + \
                                    (pSigma[out, in_idx, i] * inv_pSigma_in).reshape(3, 1) @ \
                                    (xIn[:, t] - modelPD['MuMan'][in_idx, i]).reshape(1, 1)).flatten()
                else:
                    # 如果是矩阵，使用 np.linalg.inv 计算逆矩阵
                    inv_pSigma_in = np.linalg.inv(pSigma[in_idx, in_idx, i])
                    uOut[:, i, t] = logmap_vec(modelPD['MuMan'][out, i], xhat[:, t]) + \
                                    np.dot(pSigma[out, in_idx, i], inv_pSigma_in) @ \
                                    (xIn[:, t] - modelPD['MuMan'][in_idx, i])

                # Accumulate weighted result
                uhat[:, t] += uOut[:, i, t] * H[i, t]

            # Projection back onto the manifold
            xhat[:, t] = expmap_vec(uhat[:, t], xhat[:, t]) # TODO: Debug this part to resolve the error "Singular Matrix"

        # Compute conditional covariances
        for i in range(modelPD['nbStates']):
            SigmaOutTmp = pSigma[out, out, i] - np.dot(pSigma[out, in_idx, i],
                                                       inv_pSigma_in) @ pSigma[
                              in_idx, out, i]
            expSigma[:, :, t] += H[i, t] * (SigmaOutTmp + np.outer(uOut[:, i, t], uOut[:, i, t]))

        expSigma[:, :, t] -= np.outer(uhat[:, t], uhat[:, t])
    print('GMR Regression Completed.')
    # </editor-fold>

    # <editor-fold desc="Plotting">

    #  <editor-fold desc="Plots in Cartesian space">
    # Create figure with 3 subplots for Cartesian space
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for ax in axs:
        ax.autoscale(False)
        ax.set_xlim([-15, 15])
        ax.set_ylim([-15, 15])
        ax.axhline(-15, color='black', linewidth=1.5, linestyle='-')  # 水平轴
        ax.axvline(-15, color='black', linewidth=2.5, linestyle='-')  # 垂直轴
        ax.set_xlabel(r'$x_1$', fontsize=15)
        ax.set_ylabel(r'$x_2$', fontsize=15)
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, labelsize=10)
        ax.grid(False)

    clrmap = plt.get_cmap("tab10", nbSamples)

    # --- Subplot 1: Plot demonstrations of velocity manipulability ellipsoids ---
    axs[0].set_title('Demonstrations: 2D Cartesian trajectories and manipulability ellipsoids')
    for n in range(nbSamples):

        # Plot 2D Cartesian trajectories
        axs[0].plot(s[n]['Data'][0, :], s[n]['Data'][1, :], color=[0.5, 0.5, 0.5], linewidth=2)

        for t in np.round(np.linspace(0, nbData - 1, 15)).astype(int):
            mu = np.array([s[n]['Data'][0, t], s[n]['Data'][1, t]]).reshape(2, 1)  # 提取均值
            # print("mu: ", mu,"shape: ", mu.shape)
            sigma = 1E-1 * X[1:3, 1:3, t + n * nbData]  # 缩放后的协方差矩阵
            plotGMM(mu, sigma, color=clrmap(n), axs=axs[0], valAlpha=0.4,)  # 绘制椭圆

    # --- Subplot 2: Plot GMM means ---
    axs[1].set_title('Manipulability GMM means')
    clrmap = plt.get_cmap('tab10', modelPD['nbStates'])
    # sc = 1 / modelPD['dt']

    # 绘制示教操控性椭圆（灰色，透明度较低）
    for n in range(nbSamples):
        for t in np.round(np.linspace(0, nbData - 1, 15)).astype(int):
            mu = np.array([s[n]['Data'][0, t], s[n]['Data'][1, t]]).reshape(2, 1)  # 提取均值
            # print("mu: ", mu, "shape: ", mu.shape)
            sigma = 1E-1 * X[1:3, 1:3, t + n * nbData]  # 缩放后的协方差矩阵
            plotGMM(mu, sigma, color=[0.6, 0.6, 0.6], axs=axs[1], valAlpha=0.1, edgeAlpha=0.1)  # 绘制灰色椭圆

    # 绘制每个 GMM 状态下的操控性椭圆（彩色，透明度适中）
    for i in range(modelPD['nbStates']):
        xtmp, _, _ = GMR(modelKin, modelPD['MuMan'][in_idx, i], in_idx, np.arange(1, modelKin['nbVar'])) # GMR regression
        xtmp = xtmp.reshape(2, 1)  # 将向量转换为矩阵
        sym_sigma = vec2symmat(modelPD['MuMan'][out, i])  # 将向量转换为对称矩阵
        plotGMM(xtmp, 1E-1 * sym_sigma, color=clrmap(i), axs=axs[1], valAlpha=0.4, edgeAlpha=0.3)  # 绘制彩色椭圆

    # --- Subplot 3: Plot desired reproduction ---
    axs[2].set_title('Desired reproduction')
    # 绘制期望操控性椭圆（每隔 5 个时间点绘制一次）
    for t in range(0, nbData, 5):
        # print("xd: ", xd, "shape: ", xd.shape)
        mu = np.array(xd[:, t]).reshape(2,1)  # 提取期望的均值向量
        sigma = 5E-2 * vec2symmat(xhat[:, t])  # 缩放后的协方差矩阵
        plotGMM(mu, sigma, color=[0.2, 0.8, 0.2], axs=axs[2], valAlpha=0.5, linestyle='-.', linewidth=2, edgeAlpha=1)  # 绘制绿色椭圆

    plt.tight_layout()
    plt.show()
    # </editor-fold>

    # <editor-fold desc="Time-based plots">
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)

    # 遍历 axs 中的每个子图
    for row in axs:  # 遍历二维数组的每一行
        for ax in row:  # 遍历每一行中的子图
            # ax.autoscale(False)
            ax.set_facecolor('white')
            ax.grid(False)

    clrmap = plt.get_cmap('tab10', nbSamples)

    # Plot demonstrations of velocity manipulability ellipsoids over time
    axs[0, 0].set_title('Demonstrated manipulability')

    for n in range(nbSamples):
        for t in np.round(np.linspace(0, nbData - 1, 15)).astype(int):
            plotGMM(np.array([t, 0]).reshape(2, 1), X[1:3, 1:3, t + (n * nbData)], color=clrmap(n), axs=axs[0, 0], valAlpha=0.4, edgeAlpha=0.4)

    axs[0, 0].set_xlim([-10, nbData + 10])
    axs[0, 0].set_ylim([-15, 15])
    axs[0, 0].set_ylabel(r'$\mathbf{M}$', fontsize=15)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=10)
    axs[0, 0].axhline(-15, color='black', linewidth=1, linestyle='-')  # 水平轴
    axs[0, 0].axvline(-10, color='black', linewidth=1, linestyle='-')  # 垂直轴

    # Plot demonstrated manipulability and GMM centers
    axs[1, 0].set_title('Demonstrated manipulability and GMM centers')
    sc = 1 / modelPD['dt']
    for t in range(X.shape[2]):
        plotGMM(np.array([X[in_idx, in_idx, t] * sc, 0]).reshape(2,1), X[np.ix_(outMat, outMat, [t])].squeeze(axis=2), color=[0.6, 0.6, 0.6], axs=axs[1, 0], valAlpha=0.1)

    for i in range(modelPD['nbStates']):
        sym_sigma = vec2symmat(modelPD['MuMan'][out, i])
        plotGMM(np.array([modelPD['MuMan'][in_idx, i] * sc, 0]).reshape(2,1), sym_sigma, color=clrmap(i), axs=axs[1, 0], valAlpha=0.4, edgeAlpha=0.3)

    axs[1, 0].set_xlim([-10, nbData + 10])
    axs[1, 0].set_ylim([-15, 15])
    axs[1, 0].set_xlabel(r'$t$', fontsize=15)
    axs[1, 0].set_ylabel(r'$\mathbf{M}$', fontsize=15)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=10)
    axs[1, 0].axhline(-15, color='black', linewidth=1, linestyle='-')  # 水平轴
    axs[1, 0].axvline(-10, color='black', linewidth=1, linestyle='-')  # 垂直轴

    # Plot desired manipulability profile (GMR)
    axs[0, 1].set_title('Desired manipulability profile (GMR)')
    for t in range(0, nbData, 7):
        mean = np.array([xIn[0, t] * sc, 0]).reshape(2,1)  # xIn(1,t)*sc; 0 in MATLAB
        sigma = vec2symmat(xhat[:, t])  # Convert vector to symmetric matrix
        plotGMM(mean, sigma, color=[0.2, 0.8, 0.2], axs=axs[0, 1], valAlpha=0.5, linestyle='-.', linewidth=2, edgeAlpha=1)

    # Set axis limits
    axs[0, 1].set_xlim([float(xIn[0][0]) * sc, float(xIn[0][-1]) * sc])
    axs[0, 1].set_ylim([-15, 15])

    # Set labels and font size
    axs[0, 1].set_xlabel(r'$t$', fontsize=15)
    axs[0, 1].set_ylabel(r'$\mathbf{M_d}$', fontsize=15)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=10)

    axs[0, 1].axhline(-15, color='black', linewidth=1, linestyle='-')  # 水平轴
    axs[0, 1].axvline(float(xIn[0][0]) * sc, color='black', linewidth=1, linestyle='-')  # 垂直轴

    # Plot GMM component influence
    axs[1, 1].set_title('Influence of GMM components')
    for i in range(modelPD['nbStates']):
        axs[1, 1].plot(xIn.flatten(), H[i, :], linewidth=2, color=clrmap(i))

    # Set axis limits (equivalent to MATLAB's axis)
    axs[1, 1].set_xlim([float(xIn[0][0]), float(xIn[0][-1])])
    axs[1, 1].set_ylim([0, 1.02])

    # Set labels and font sizes
    axs[1, 1].set_xlabel(r'$t$', fontsize=15)
    axs[1, 1].set_ylabel(r'$h_k$', fontsize=15)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=10)
    axs[1, 1].axhline(0, color='black', linewidth=1, linestyle='-')  # 水平轴
    axs[1, 1].axvline(float(xIn[0][0]), color='black', linewidth=1, linestyle='-')  # 垂直轴

    # Show the plot
    plt.tight_layout()
    plt.show()
    # </editor-fold>

    ## <editor-fold desc="Plots in the SPD space">
    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(131, projection='3d'),
            fig.add_subplot(132, projection='3d'),
            fig.add_subplot(133, projection='3d')]
    clrmap = plt.get_cmap('tab10', nbSamples)  # Color map for different samples

    # Create cone data
    xax, yax, zax = create_cone_data()
    direction = np.cross([1, 0, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0])  # Rotation axis: Positive direction of Z-axis

    # Subplot 1: Demonstrations
    ax = axes[0]
    ax.set_title(r"Demonstrations: manipulability ellipsoids", fontsize=12)
    plot_cone(ax, xax, yax, zax, direction, 65)

    # Plot manipulability ellipsoids
    for n in range(nbSamples):
        for t in range(nbData):
            ax.plot([x[1, t + n * nbData]], [x[2, t + n * nbData]], [x[3, t + n * nbData] / np.sqrt(2)],
                    '.', markersize=6, color=clrmap(n))

    # Subplot 2: GMM in SPD space
    ax = axes[1]
    ax.set_title(r"Manipulability GMM in SPD space", fontsize=12)
    plot_cone(ax, xax, yax, zax, direction, 65)

    # Plot manipulability ellipsoids
    for n in range(nbSamples):
        for t in range(nbData):
            ax.plot([x[1, t + n * nbData]], [x[2, t + n * nbData]], [x[3, t + n * nbData] / np.sqrt(2)],
                    '.', markersize=6, color=[0.6, 0.6, 0.6])

    # Plot GMM ellipsoids
    for i in range(modelPD['nbStates']):
        mu = np.array(modelPD['MuMan'][out, i]).reshape(3,1)  # Extract mean for the current state
        mu[2] = mu[2] / np.sqrt(2)  # Rescale the 3rd dimension for plotting
        sigma = modelPD['Sigma'][np.ix_(out, out, [i])].squeeze(axis=2)  # Extract covariance matrix for the current state
        sigma[2, :] /= np.sqrt(2)  # Rescale the 3rd row of the covariance matrix
        sigma[:, 2] /= np.sqrt(2)  # Rescale the 3rd column of the covariance matrix

        sigma += 5 * np.eye(3)  # Add small identity for better visualization

        # Plot the Gaussian with its covariance matrix as an ellipsoid
        plotGMM3D(mu, sigma, color=clrmap(i), axs=ax, alpha=0.6)

    # Subplot 3: Desired reproduction
    ax = axes[2]
    ax.set_title(r"Desired reproduction", fontsize=12)
    plot_cone(ax, xax, yax, zax, direction, 65)
    # Plot reproduced manipulability ellipsoid
    ax.plot(xhat[0, :], xhat[1, :], xhat[2, :] / np.sqrt(2), '.', markersize=6, color=[0.2, 0.8, 0.2])

    # Set axis off and adjust view
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()
        ax.view_init(elev=12, azim=70)

    plt.tight_layout()
    plt.show()
    # </editor-fold>

    # </editor-fold>

if __name__ == "__main__":
    ManipulabilityLearning()