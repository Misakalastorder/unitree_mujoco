import math
import time
import threading
import json
from threading import Thread

import mujoco
import mujoco.viewer
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.rpc.server import Server
from unitree_sdk2py.rpc.internal import RPC_ERR_SERVER_API_PARAMETER
from unitree_sdk2py.g1.loco.g1_loco_api import (
    LOCO_SERVICE_NAME,
    LOCO_API_VERSION,
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
    ROBOT_API_ID_LOCO_GET_BALANCE_MODE,
    ROBOT_API_ID_LOCO_GET_SWING_HEIGHT,
    ROBOT_API_ID_LOCO_GET_STAND_HEIGHT,
    ROBOT_API_ID_LOCO_GET_PHASE,
    ROBOT_API_ID_LOCO_SET_FSM_ID,
    ROBOT_API_ID_LOCO_SET_BALANCE_MODE,
    ROBOT_API_ID_LOCO_SET_SWING_HEIGHT,
    ROBOT_API_ID_LOCO_SET_STAND_HEIGHT,
    ROBOT_API_ID_LOCO_SET_VELOCITY,
    ROBOT_API_ID_LOCO_SET_ARM_TASK,
)
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
        self.fsm_id = 200
        self.fsm_mode = "stand"
        self.balance_mode = 1
        self.swing_height = 0.0
        self.stand_height = 1.0
        self.arm_task_id = 0
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

        self.attitude_kp = 6.0
        self.attitude_kd = 0.6
        self.max_roll_pitch_correction = 0.12
        self.ramp_time = 4.0
        self.stand_enter_time = self.start_time
        self.damp_kd = 2.0

    def _apply_stance_bias(self):
        # 在中立姿态上叠加站立偏置，形成更稳定的目标姿态。
        self.stance_offsets = {
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

        for joint_name, offset in self.stance_offsets.items():
            self.stand_pose[self.joint_index[joint_name]] = (
                self.neutral_pose[self.joint_index[joint_name]] + offset
            )

    def _update_pose_by_stand_height(self):
        # 站高越高腿部弯曲越小，站高越低腿部弯曲越大。
        self.stand_pose = self.neutral_pose.copy()
        height = float(np.clip(self.stand_height, 0.0, 1.0))
        leg_scale = 1.25 - 0.50 * height
        for joint_name, offset in self.stance_offsets.items():
            scale = leg_scale if ("hip" in joint_name or "knee" in joint_name or "ankle" in joint_name) else 1.0
            self.stand_pose[self.joint_index[joint_name]] = (
                self.neutral_pose[self.joint_index[joint_name]] + offset * scale
            )

    def set_fsm_id(self, fsm_id):
        self.fsm_id = int(fsm_id)
        if self.fsm_id == 0:
            self.fsm_mode = "zero_torque"
        elif self.fsm_id == 1:
            self.fsm_mode = "damp"
        else:
            self.fsm_mode = "stand"
            self.stand_enter_time = time.perf_counter()

    def set_balance_mode(self, balance_mode):
        self.balance_mode = int(balance_mode)

    def set_swing_height(self, swing_height):
        self.swing_height = float(swing_height)

    def set_stand_height_raw(self, stand_height_raw):
        raw = float(stand_height_raw)
        if raw > 1.0:
            raw = raw / float((1 << 32) - 1)
        self.stand_height = float(np.clip(raw, 0.0, 1.0))
        self._update_pose_by_stand_height()

    def set_arm_task(self, task_id):
        self.arm_task_id = int(task_id)

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
        if self.fsm_mode == "zero_torque":
            self.data.ctrl[: self.model.nu] = 0.0
            return

        q = np.array(
            [self.data.qpos[self.joint_qpos_addr[name]] for name in self.JOINT_NAMES],
            dtype=float,
        )
        dq = np.array(
            [self.data.qvel[self.joint_dof_addr[name]] for name in self.JOINT_NAMES],
            dtype=float,
        )

        if self.fsm_mode == "damp":
            torque = -self.damp_kd * dq
            self.data.ctrl[: self.model.nu] = np.clip(torque, self.ctrl_lower, self.ctrl_upper)
            return

        elapsed = time.perf_counter() - self.stand_enter_time
        blend = min(1.0, elapsed / self.ramp_time)

        desired = (1.0 - blend) * self.neutral_pose + blend * self.stand_pose

        roll, pitch = self._torso_roll_pitch()
        angular_velocity = self.data.qvel[3:6]
        if self.balance_mode != 0:
            roll_correction = -self.attitude_kp * roll - self.attitude_kd * angular_velocity[0]
            pitch_correction = -self.attitude_kp * pitch - self.attitude_kd * angular_velocity[1]
        else:
            roll_correction = 0.0
            pitch_correction = 0.0

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


class SimLocoServer(Server):
    # 在仿真进程内实现最小 loco RPC 服务，便于直接用 LocoClient 初始化站立。
    def __init__(self, stand_controller):
        super().__init__(LOCO_SERVICE_NAME)
        self.controller = stand_controller
        self.phase = 0.0

    def Init(self):
        self._SetApiVersion(LOCO_API_VERSION)
        self._RegistHandler(ROBOT_API_ID_LOCO_GET_FSM_ID, self.GetFsmId, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_GET_FSM_MODE, self.GetFsmMode, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_GET_BALANCE_MODE, self.GetBalanceMode, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_GET_SWING_HEIGHT, self.GetSwingHeight, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_GET_STAND_HEIGHT, self.GetStandHeight, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_GET_PHASE, self.GetPhase, 0)

        self._RegistHandler(ROBOT_API_ID_LOCO_SET_FSM_ID, self.SetFsmId, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_SET_BALANCE_MODE, self.SetBalanceMode, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_SET_SWING_HEIGHT, self.SetSwingHeight, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_SET_STAND_HEIGHT, self.SetStandHeight, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_SET_VELOCITY, self.SetVelocity, 0)
        self._RegistHandler(ROBOT_API_ID_LOCO_SET_ARM_TASK, self.SetArmTask, 0)

    def _parse_parameter(self, parameter):
        try:
            return json.loads(parameter)
        except Exception:
            return None

    def _ok_data(self, value):
        return 0, json.dumps({"data": value})

    def SetFsmId(self, parameter):
        payload = self._parse_parameter(parameter)
        if payload is None or "data" not in payload:
            return RPC_ERR_SERVER_API_PARAMETER, ""
        locker.acquire()
        try:
            self.controller.set_fsm_id(int(payload["data"]))
        finally:
            locker.release()
        return 0, ""

    def SetBalanceMode(self, parameter):
        payload = self._parse_parameter(parameter)
        if payload is None or "data" not in payload:
            return RPC_ERR_SERVER_API_PARAMETER, ""
        locker.acquire()
        try:
            self.controller.set_balance_mode(int(payload["data"]))
        finally:
            locker.release()
        return 0, ""

    def SetSwingHeight(self, parameter):
        payload = self._parse_parameter(parameter)
        if payload is None or "data" not in payload:
            return RPC_ERR_SERVER_API_PARAMETER, ""
        locker.acquire()
        try:
            self.controller.set_swing_height(float(payload["data"]))
        finally:
            locker.release()
        return 0, ""

    def SetStandHeight(self, parameter):
        payload = self._parse_parameter(parameter)
        if payload is None or "data" not in payload:
            return RPC_ERR_SERVER_API_PARAMETER, ""
        locker.acquire()
        try:
            self.controller.set_stand_height_raw(payload["data"])
        finally:
            locker.release()
        return 0, ""

    def SetVelocity(self, parameter):
        # 当前示例只处理站立初始化，速度命令仅接受并忽略。
        payload = self._parse_parameter(parameter)
        if payload is None:
            return RPC_ERR_SERVER_API_PARAMETER, ""
        return 0, ""

    def SetArmTask(self, parameter):
        payload = self._parse_parameter(parameter)
        if payload is None or "data" not in payload:
            return RPC_ERR_SERVER_API_PARAMETER, ""
        locker.acquire()
        try:
            self.controller.set_arm_task(payload["data"])
        finally:
            locker.release()
        return 0, ""

    def GetFsmId(self, parameter):
        locker.acquire()
        try:
            return self._ok_data(self.controller.fsm_id)
        finally:
            locker.release()

    def GetFsmMode(self, parameter):
        mode_map = {"zero_torque": 0, "damp": 1, "stand": 2}
        locker.acquire()
        try:
            return self._ok_data(mode_map.get(self.controller.fsm_mode, 2))
        finally:
            locker.release()

    def GetBalanceMode(self, parameter):
        locker.acquire()
        try:
            return self._ok_data(self.controller.balance_mode)
        finally:
            locker.release()

    def GetSwingHeight(self, parameter):
        locker.acquire()
        try:
            return self._ok_data(self.controller.swing_height)
        finally:
            locker.release()

    def GetStandHeight(self, parameter):
        locker.acquire()
        try:
            return self._ok_data(self.controller.stand_height)
        finally:
            locker.release()

    def GetPhase(self, parameter):
        return self._ok_data(self.phase)


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
    loco_server = SimLocoServer(g1_stand_controller)
    loco_server.Init()
    loco_server.Start(False)
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
