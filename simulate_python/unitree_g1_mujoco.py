import math
import time
import threading
from threading import Thread

import mujoco
import mujoco.viewer
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()
# 全局互斥锁：保护仿真线程与可视化线程对 mj_data 的并发访问。
 
mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
# 加载 MuJoCo 模型：从配置指定的场景 XML 创建模型对象。
mj_data = mujoco.MjData(mj_model)
# 创建运行时数据容器：保存状态、控制输入和中间计算结果。


def _joint_qpos_addr(model, joint_name):
    # 查询关节在 qpos 向量中的起始索引，用于读取关节位置。
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[joint_id]


def _joint_dof_addr(model, joint_name):
    # 查询关节在 qvel 向量中的自由度索引，用于读取关节速度。
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_dofadr[joint_id]


class G1StandController:
    # G1 站立控制器：生成关节 PD 扭矩并叠加躯干姿态补偿。
    JOINT_NAMES = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    def __init__(self, model, data):
        # 初始化控制器参数、关节映射、目标位姿与增益配置。
        self.model = model
        self.data = data
        self.start_time = time.perf_counter()

        self.joint_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.JOINT_NAMES
        }
        self.joint_qpos_addr = {
            name: _joint_qpos_addr(model, name) for name in self.JOINT_NAMES
        }
        self.joint_dof_addr = {
            name: _joint_dof_addr(model, name) for name in self.JOINT_NAMES
        }
        self.joint_index = {name: index for index, name in enumerate(self.JOINT_NAMES)}

        self.neutral_pose = np.array(
            [self.data.qpos[address] for address in self.joint_qpos_addr.values()],
            dtype=float,
        )
        self.stand_pose = self.neutral_pose.copy()
        self._apply_stance_bias()

        ctrlrange = np.asarray(self.model.actuator_ctrlrange[: self.model.nu], dtype=float)
        self.ctrl_lower = ctrlrange[:, 0]
        self.ctrl_upper = ctrlrange[:, 1]

        self.position_gains = np.array(
            [self._joint_position_gain(name) for name in self.JOINT_NAMES], dtype=float
        )
        self.velocity_gains = np.array(
            [self._joint_velocity_gain(name) for name in self.JOINT_NAMES], dtype=float
        )

        self.pose_lower = np.array(
            [
                self.model.jnt_range[self.joint_ids[name], 0]
                if self.model.jnt_limited[self.joint_ids[name]]
                else -np.inf
                for name in self.JOINT_NAMES
            ],
            dtype=float,
        )
        self.pose_upper = np.array(
            [
                self.model.jnt_range[self.joint_ids[name], 1]
                if self.model.jnt_limited[self.joint_ids[name]]
                else np.inf
                for name in self.JOINT_NAMES
            ],
            dtype=float,
        )

        self.attitude_kp = 50
        self.attitude_kd = 5
        self.max_roll_pitch_correction = 0.12
        self.ramp_time = 4.0

    def _apply_stance_bias(self):
        # 在中立姿态上叠加站立偏置，形成更稳定的目标姿态。
        stance_offsets = {
            "left_hip_pitch_joint": -0.18,
            "left_knee_joint": 0.40,
            "left_ankle_pitch_joint": -0.22,
            "right_hip_pitch_joint": -0.18,
            "right_knee_joint": 0.40,
            "right_ankle_pitch_joint": -0.22,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.10,
            "left_shoulder_roll_joint": 0.15,
            "left_elbow_joint": 0.20,
            "right_shoulder_pitch_joint": 0.10,
            "right_shoulder_roll_joint": -0.15,
            "right_elbow_joint": 0.20,
        }

        for joint_name, offset in stance_offsets.items():
            self.stand_pose[self.joint_index[joint_name]] = (
                self.neutral_pose[self.joint_index[joint_name]] + offset
            )

    def _joint_position_gain(self, joint_name):
        # 按关节类型返回位置环增益 Kp。
        if "ankle" in joint_name:
            return 70.0
        if "knee" in joint_name:
            return 80.0
        if "hip" in joint_name:
            return 60.0
        if "waist" in joint_name:
            return 55.0
        if "shoulder" in joint_name or "elbow" in joint_name:
            return 25.0
        if "wrist" in joint_name:
            return 12.0
        return 30.0

    def _joint_velocity_gain(self, joint_name):
        # 按关节类型返回速度环增益 Kd。
        if "ankle" in joint_name:
            return 3.0
        if "knee" in joint_name:
            return 3.5
        if "hip" in joint_name:
            return 2.8
        if "waist" in joint_name:
            return 2.4
        if "shoulder" in joint_name or "elbow" in joint_name:
            return 1.2
        if "wrist" in joint_name:
            return 0.7
        return 1.6

    def _body_rotation(self, body_name):
        # 读取指定刚体的旋转矩阵，兼容不同 xmat 内存布局。
        body_id = self.model.body(body_name).id
        xmat = np.asarray(self.data.xmat, dtype=float)

        if xmat.ndim == 2 and xmat.shape[0] > body_id and xmat.shape[1] >= 9:
            return xmat[body_id, :9].reshape(3, 3)

        flat = xmat.reshape(-1)
        start = body_id * 9
        if flat.size >= start + 9:
            return flat[start : start + 9].reshape(3, 3)

        return np.eye(3)

    def _torso_roll_pitch(self):
        # 基于躯干旋转矩阵计算 roll 与 pitch 欧拉角。
        rotation = self._body_rotation("torso_link")
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], math.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2))
        return roll, pitch

    def apply(self):
        # 主控制入口：插值目标位姿、姿态补偿、限幅后写入扭矩控制。
        elapsed = time.perf_counter() - self.start_time
        blend = min(1.0, elapsed / self.ramp_time)

        q = np.array(
            [self.data.qpos[self.joint_qpos_addr[name]] for name in self.JOINT_NAMES],
            dtype=float,
        )
        dq = np.array(
            [self.data.qvel[self.joint_dof_addr[name]] for name in self.JOINT_NAMES],
            dtype=float,
        )

        desired = (1.0 - blend) * self.neutral_pose + blend * self.stand_pose

        roll, pitch = self._torso_roll_pitch()
        angular_velocity = self.data.qvel[3:6]
        roll_correction = -self.attitude_kp * roll - self.attitude_kd * angular_velocity[0]
        pitch_correction = -self.attitude_kp * pitch - self.attitude_kd * angular_velocity[1]

        roll_correction = float(
            np.clip(
                roll_correction,
                -self.max_roll_pitch_correction,
                self.max_roll_pitch_correction,
            )
        )
        pitch_correction = float(
            np.clip(
                pitch_correction,
                -self.max_roll_pitch_correction,
                self.max_roll_pitch_correction,
            )
        )

        desired[self.joint_index["waist_roll_joint"]] += 0.10 * roll_correction
        desired[self.joint_index["waist_pitch_joint"]] += 0.10 * pitch_correction

        desired[self.joint_index["left_hip_roll_joint"]] += 0.06 * roll_correction
        desired[self.joint_index["right_hip_roll_joint"]] += 0.06 * roll_correction
        desired[self.joint_index["left_ankle_roll_joint"]] += 0.09 * roll_correction
        desired[self.joint_index["right_ankle_roll_joint"]] += 0.09 * roll_correction

        desired[self.joint_index["left_hip_pitch_joint"]] += 0.07 * pitch_correction
        desired[self.joint_index["right_hip_pitch_joint"]] += 0.07 * pitch_correction
        desired[self.joint_index["left_ankle_pitch_joint"]] += 0.11 * pitch_correction
        desired[self.joint_index["right_ankle_pitch_joint"]] += 0.11 * pitch_correction

        desired = np.clip(desired, self.pose_lower, self.pose_upper)

        torque = self.position_gains * (desired - q) - self.velocity_gains * dq
        torque = np.clip(torque, self.ctrl_lower, self.ctrl_upper)
        self.data.ctrl[: self.model.nu] = torque


if config.ENABLE_ELASTIC_BAND:
    # 启用弹力带模式：注册键盘回调并准备对机体施加外力。
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    # 普通模式：仅启动被动可视化窗口，不加载弹力带交互。
    # viewer = mujoco.viewer.launch(mj_model, mj_data)
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
# 设置仿真步长：控制物理积分与主循环节拍。
num_motor_ = mj_model.nu
# 电机数量：等于模型执行器数量。
dim_motor_sensor_ = 3 * num_motor_
# 电机观测维度占位：每个电机默认三类传感量。
g1_stand_controller = G1StandController(mj_model, mj_data)
# 创建站立控制器实例。

time.sleep(0.2)


def SimulationThread():
    # 仿真线程：处理通信初始化、控制计算、物理步进与可选外力。
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    mujoco.mj_forward(mj_model, mj_data)

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        g1_stand_controller.apply()
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    # 可视化线程：同步 viewer 显示，避免阻塞仿真步进。
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    # 程序入口：并行启动可视化线程和仿真线程。
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
