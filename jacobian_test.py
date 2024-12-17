# import numpy as np
# from roboticstoolbox import DHRobot, RevoluteDH
#
# # 定义机器人模型
# L1 = RevoluteDH(a=1, alpha=0)
# L2 = RevoluteDH(a=1, alpha=0)
# left_arm = DHRobot([L1, L2], name="2DOF Robot")
#
# # 关节状态
# q_left = np.array([-2.0461, 1.8690])
#
# # 计算雅可比矩阵
# J_left = left_arm.jacob0(q_left)[:3, :]  # 提取平移部分
# print(J_left)

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

# 创建 2DOF 机器人的 URDF 文件（与 MATLAB 的模型一致）
urdf_content = """
<?xml version="1.0" ?>
<robot name="2DOF Robot">
  <link name="base"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="end_effector"/>
  
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>

  <joint name="joint3" type="fixed">
    <parent link="link2"/>
    <child link="end_effector"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""

# 保存 URDF 文件
urdf_path = "data/urdf/2dof_robot.urdf"
with open(urdf_path, "w") as f:
    f.write(urdf_content)


# 清理 URDF 文件内容
def clean_urdf_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 去除空行和无效字符
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))


# 清理文件后再加载
clean_urdf_file(r"/data/urdf/2dof_robot.urdf")
robot = rtb.ERobot.URDF(r"/data/urdf/2dof_robot.urdf")
print("Robot loaded:", robot)

# 定义关节状态（与 MATLAB 中一致）
q = np.array([-2.0461, 1.8690])

# 计算雅可比矩阵
J = robot.jacob0(q)
print("Jacobian:", J)

# 提取平移部分（前 3 行）
J_trans = J[:3, :]
print("Translational Jacobian (Python):")
print(J_trans)