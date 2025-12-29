# # 文档版
# import math
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import os
# import gym
# from gym.spaces import Box

# # ------------------- 常量参数定义（含单位说明） -------------------
# AREA_SIZE = 500.0  # m，环境边长
# HALF_AREA = AREA_SIZE / 2.0  # m, 半边长
# DT = 1.0  # s，每时间步长度

# NUM_UAV = 3  # 无人机数量
# NUM_TARGET = 20  # 目标数量
# NUM_GN = 10  # 地面设备数量

# # 无人机起始位置（m），从 [0, 500] 移到 [-250, 250]
# UAV_START_POS_ORIGINAL = np.array([[50.0, 50.0], [350.0, 100.0], [200.0, 400.0]])
# UAV_START_POS = UAV_START_POS_ORIGINAL - HALF_AREA

# # 高度参数
# H_UAV = 15.0  # m，无人机高度
# H_GN = 5.0  # m，地面设备高度
# H_TARGET = 0.0  # m，目标高度

# # 运动限制
# D_MAX = 40.0  # m，每步最大移动距离 (最大欧氏距离)
# SAFE_DISTANCE = 40.0  # m，无人机最小安全距离

# # 通信与感知参数
# P_TX = 30.0  # W，无人机发射功率
# B = 2e6  # Hz，通信带宽
# FC = 3e9  # Hz，载波频率
# C = 3e8  # m/s，光速
# LAMBDA = C / FC  # m，波长
# G_TX = 10 ** 1.3  # 无量纲，发射天线增益
# G_RX = 10 ** 1.3  # 无量纲，接收天线增益
# SIGMA_RCS = 0.25  # m²，雷达散射截面

# KB = 1.38e-23  # J/K，玻尔兹曼常数
# TEMP_K = 290.0  # K，噪声温度
# NOISE_FIG_DB = 10 ** 0.5  # dB，噪声系数
# N0 = KB * TEMP_K * B * NOISE_FIG_DB  # 噪声功率

# # 信道参数
# LOS_A = 4.88
# LOS_B = 0.43
# ATT_LOS = 0.1  # LoS 衰减系数
# ATT_NLOS = 21  # NLoS 衰减系数
# ZIP0 = 10 ** -20.4  # 噪声dBm/Hz

# # 能量收集参数（文档公式）
# EH_U = 24e-3
# EH_r1 = 150
# EH_r2 = 0.014

# # GN 能量模型参数
# GN_E_MAX = 50.0  # mJ，地面设备最大能量
# GN_E_INIT = 20.0  # mJ，初始能量
# GN_BASE_CONSUME_MEAN = 0.08  # mJ/s，平均耗能速率
# GN_BASE_CONSUME_STD = 0.02  # mJ/s，耗能标准差

# # UAV 飞行能耗模型参数
# V_TIP = 120.0  # m/s，旋翼叶尖速度
# C1 = 80.0  # W，桨叶剖面功率系数
# C2 = 22.0
# C3 = 263.4  # W，诱导功率系数
# C4 = 0.0092  # 无量纲，寄生功率系数

# # 奖励权重与惩罚
# W_SENSE = 10.0
# W_CHARGE = 8
# W_ENERGY = 0.5
# W_PENALTY = 50.0
# COLLISION_PENALTY = 20.0
# OUT_OF_BOUNDS_PENALTY = 20.0
# W1 = 1000  # 首次探测的奖励
# α = 0.1  # 衰减因子
# W2 = 300  # 充电激励奖励

# GN_ENERGY_THRESHOLD = 13.5  # mJ
# MIN_SINR_DB = 15.0  # dB，探测 SINR 阈值
# MIN_RATE = 1e5  # bps，通信速率阈值
# THETA_HALF_BW_DEG_GN = 60.0  # 通信范围


# # 随机种子
# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)


# # ------------------- 工具函数 -------------------
# # 动作空间 a \in [-1, 1]
# def angle_from_norm(theta_norm):
#     """角度：[-1, 1] -> [-pi, pi]"""
#     return float(theta_norm) * math.pi


# def dist_from_norm(dist_norm):
#     """距离：[-1, 1] -> [-D_MAX, D_MAX]"""
#     return float(dist_norm) * D_MAX


# def beta_from_norm(beta_norm):
#     """beta: [-1, 1] -> [0, 1]"""
#     return (float(beta_norm) + 1.0) / 2.0


# def rho_from_norm(rho_norm):
#     """rho: [-1, 1] -> [0, 1]"""
#     return (float(rho_norm) + 1.0) / 2.0


# def los_probability(uav_pos, gn_pos):
#     """LoS 概率模型（文档公式）"""
#     dx = gn_pos[0] - uav_pos[0]
#     dy = gn_pos[1] - uav_pos[1]
#     horiz = math.hypot(dx, dy)
#     elev = math.atan2(H_UAV - H_GN, horiz)
#     elev_deg = math.degrees(elev)
#     p = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (elev_deg - LOS_A)))
#     return p


# def avg_pathloss_linear(uav_pos, gn_pos, theta_half_bw_deg=60.0):
#     """平均信道功率增益（含定向天线影响）"""
#     dx = gn_pos[0] - uav_pos[0]
#     dy = gn_pos[1] - uav_pos[1]
#     d3d = math.hypot(math.hypot(dx, dy), H_UAV - H_GN)
#     p_l = los_probability(uav_pos, gn_pos)
#     eta_avg = (d3d ** -2) * ((4 * math.pi * FC / C) ** -2) * ((p_l * ATT_LOS + (1 - p_l) * ATT_NLOS) ** -1)
#     theta_rad = math.radians(theta_half_bw_deg)
#     horiz = math.hypot(dx, dy)
#     if horiz <= (H_UAV - H_GN) * math.tan(theta_rad):
#         dir_gain_factor = 2.28 / (theta_rad ** 2)
#         h = eta_avg * dir_gain_factor
#     else:
#         h = eta_avg
#     return h


# def radar_received_power(uav_pos, target_pos, p_side):
#     """雷达回波功率（文档公式）"""
#     dx = target_pos[0] - uav_pos[0]
#     dy = target_pos[1] - uav_pos[1]
#     R = math.hypot(math.hypot(dx, dy), H_UAV - H_TARGET)
#     Pr = p_side * G_TX * G_RX * (LAMBDA ** 2 * SIGMA_RCS) / ((4.0 * math.pi) ** 3 * (R ** 4))
#     return Pr


# def comm_rx_power(uav_pos, gn_pos, p_main, rho):
#     """通信接收功率 = ρ * p_main * g"""
#     g = avg_pathloss_linear(uav_pos, gn_pos)
#     P_rx = rho * p_main * g
#     return float(max(P_rx, 0.0))


# def rf_energy_harvested(p_rx):
#     """能量收集电路模型（文档公式）"""
#     val = EH_U / (1 + math.exp(EH_r1 * EH_r2)) * ((1 + math.exp(EH_r1 * EH_r2)) /
#                                                   (1 + math.exp(EH_r1 * EH_r2) * math.exp(-EH_r1 * p_rx)) - 1)
#     return val


# def uav_flight_energy(v):
#     """无人机能耗（文档物理模型）"""
#     # 注意：v 是速度的绝对值
#     term1 = C1 * (1.0 + 3.0 * (v ** 2) / (V_TIP ** 2))
#     term2 = C2 * math.sqrt(math.sqrt(C3 + (v ** 4) / 4) - (v ** 2) / 2)
#     term3 = C4 * (v ** 3)
#     P_total = term1 + term2 + term3
#     return P_total * DT


# def lin2db(x):
#     return 10.0 * math.log10(max(x, 1e-30))


# def jain_index(x):
#     x = np.array(x, dtype=float)
#     num = (np.sum(x) ** 2)
#     den = len(x) * np.sum(x ** 2) + 1e-12
#     return float(num / den)


# # ------------------- 环境定义 -------------------
# class UAVEnvFixed:
#     def __init__(self,
#                  target_pos_list=None,
#                  gn_pos_list=None,
#                  gn_energy_init=GN_E_INIT,
#                  episode_steps=80):
#         self.area = AREA_SIZE
#         self.half_area = HALF_AREA  # [-250, 250]
#         self.episode_steps = episode_steps

#         # 原始固定位置 (在 [0, 500])
#         target_pos_original = np.array([
#             [50.0, 120.0], [200.0, 180.0], [300.0, 60.0], [400.0, 350.0], [120.0, 450.0],
#             [250.0, 400.0], [60.0, 320.0], [470.0, 200.0], [100.0, 280.0], [370.0, 450.0],
#             [50.0, 100.0], [420.0, 150.0], [150.0, 240.0], [320.0, 50.0], [380.0, 120.0],
#             [100.0, 400.0], [270.0, 150.0], [220.0, 220.0], [60.0, 250.0], [420.0, 70.0]
#         ])

#         gn_pos_original = np.array([
#             [30.0, 220.0], [160.0, 180.0], [240.0, 30.0], [370.0, 400.0], [50.0, 90.0],
#             [310.0, 330.0], [320.0, 180.0], [440.0, 270.0], [180.0, 440.0], [90.0, 400.0]
#         ])

#         # 转换为 [-250, 250] 坐标系
#         self.target_pos = (target_pos_original - self.half_area) if target_pos_list is None else (
#                 np.array(target_pos_list) - self.half_area)
#         self.gn_pos = (gn_pos_original - self.half_area) if gn_pos_list is None else (
#                 np.array(gn_pos_list) - self.half_area)

#         # 状态初始化
#         self.uav_pos = UAV_START_POS.copy()
#         self.uav_prev_pos = self.uav_pos.copy()
#         self.uav_prev_actions = np.zeros((NUM_UAV, 4))  # 存储的是归一化后的动作 [-1, 1]
#         self.gn_energy = np.ones(NUM_GN) * gn_energy_init
#         self.target_detect_count = np.zeros(NUM_TARGET, dtype=int)
#         self.t = 0
#         self.agent_paths = [[] for _ in range(NUM_UAV)]  # 用于存储每个无人机的轨迹

#         # --- 新增：为render函数存储历史探测数据 ---
#         self.detection_history = []

#         # --- 为 MARL 框架添加的属性 ---
#         self.agent_num = NUM_UAV
#         self.uav_num = NUM_UAV  # 兼容 PPO 和 MAAS

#         # 观测维度 = 6 (xn, yn, prev_dist, prev_theta, prev_beta, prev_rho)
#         self.observation_space_len = 6
#         # 动作维度 = 4 (theta, dist, beta, rho)
#         self.action_space_len = 4

#         # 定义 Gym 空间 (PPO 和 DDPG/SAC 的 agents_config_dict 需要)
#         # 观测空间 [-1, 1]，注意形状是 (obs_len,)
#         self.observation_space = Box(-1.0, 1.0, (self.observation_space_len,))
#         # 动作空间 [-1, 1]，注意形状是 (act_len,)
#         self.action_space = Box(-1.0, 1.0, (self.action_space_len,))
#         # --- MARL 属性添加完毕 ---

#         # 轨迹初始化 (UAV_START_POS 已经在 [-250, 250] 范围内)
#         for i in range(NUM_UAV):
#             self.agent_paths[i].append(self.uav_pos[i].copy())

#     def reset(self):
#         self.uav_pos = UAV_START_POS.copy()
#         self.uav_prev_pos = self.uav_pos.copy()
#         self.uav_prev_actions.fill(0.0)
#         self.gn_energy = np.ones(NUM_GN) * GN_E_INIT
#         self.target_detect_count[:] = 0
#         self.t = 0

#         # --- 新增：重置历史记录 ---
#         self.detection_history = []

#         # 初始化 UAV 的轨迹
#         self.agent_paths = [[] for _ in range(NUM_UAV)]
#         for i in range(NUM_UAV):
#             self.agent_paths[i].append(self.uav_pos[i].copy())

#         return self._get_obs()

#     def _get_obs(self):
#         obs = []
#         for i in range(NUM_UAV):
#             x, y = self.uav_pos[i]
#             # 归一化到 [-1, 1] 范围
#             xn, yn = x / self.half_area, y / self.half_area
#             prev = self.uav_prev_actions[i]
#             # prev[0]=theta, prev[1]=dist, prev[2]=beta, prev[3]=rho (都是归一化的)
#             s = np.array([xn, yn, prev[1], prev[0], prev[2], prev[3]], dtype=np.float32)
#             obs.append(s)
#         return np.stack(obs, axis=0)

#     def step(self, actions):
#         # 动作 a \in [-1, 1]
#         actions = np.clip(np.array(actions, dtype=float), -1.0, 1.0)

#         # 实际动作 (real_actions) 存储转换后的物理值：[theta_rad, d_m, beta_[0,1], rho_[0,1]]
#         real_actions = np.zeros_like(actions)
#         for i in range(NUM_UAV):
#             theta = angle_from_norm(actions[i, 0])  # [-pi, pi]
#             d = dist_from_norm(actions[i, 1])  # [-D_MAX, D_MAX]
#             beta = beta_from_norm(actions[i, 2])  # [0, 1]
#             rho = rho_from_norm(actions[i, 3])  # [0, 1]
#             real_actions[i] = [theta, d, beta, rho]

#         # UAV 移动
#         self.uav_prev_pos = self.uav_pos.copy()
#         new_positions = self.uav_pos.copy()
#         uav_speeds = np.zeros(NUM_UAV)
#         for i in range(NUM_UAV):
#             theta, d = real_actions[i, 0], real_actions[i, 1]
#             dx = d * math.cos(theta)
#             dy = d * math.sin(theta)
#             nx = self.uav_pos[i, 0] + dx
#             ny = self.uav_pos[i, 1] + dy
#             new_positions[i] = [nx, ny]
#             # 速度 v = |d| / DT
#             uav_speeds[i] = abs(d) / DT
#         self.uav_pos = new_positions.copy()

#         # 违规检测
#         collision_flags = np.zeros(NUM_UAV, dtype=int)  # 碰撞
#         oob_flags = np.zeros(NUM_UAV, dtype=int)  # 越界

#         lower_bound = -self.half_area
#         upper_bound = self.half_area

#         for i in range(NUM_UAV):
#             x, y = self.uav_pos[i]
#             # 越界检测: 范围 [-250, 250]
#             if x < lower_bound or x > upper_bound or y < lower_bound or y > upper_bound:
#                 oob_flags[i] = 1
#                 self.uav_pos[i] = self.uav_prev_pos[i].copy()

#         for i in range(NUM_UAV):
#             for j in range(i + 1, NUM_UAV):
#                 di = np.linalg.norm(self.uav_pos[i] - self.uav_pos[j])
#                 if di < SAFE_DISTANCE:
#                     collision_flags[i] = collision_flags[j] = 1
#                     # 避免在同一时间步内多次回退
#                     if oob_flags[i] == 0: self.uav_pos[i] = self.uav_prev_pos[i].copy()
#                     if oob_flags[j] == 0: self.uav_pos[j] = self.uav_prev_pos[j].copy()

#         # GN 分配（基于定向天线约束）
#         chosen_gn = -np.ones(NUM_UAV, dtype=int)
#         gn_taken = np.zeros(NUM_GN, dtype=bool)

#         THETA_HALF_BW_RAD_GN = math.radians(THETA_HALF_BW_DEG_GN)
#         MAX_BEAM_HORIZ_DIST_GN = (H_UAV - H_GN) * math.tan(THETA_HALF_BW_RAD_GN)

#         # if self.t == 0:
#         #     print(f"[Env Info] Max horizontal distance for GN beam: {MAX_BEAM_HORIZ_DIST_GN:.2f} m")

#         for i in range(NUM_UAV):
#             dists = np.linalg.norm(self.gn_pos - self.uav_pos[i], axis=1)
#             order = np.argsort(dists)

#             for g in order:
#                 is_available = not gn_taken[g]
#                 is_in_beam = (dists[g] <= MAX_BEAM_HORIZ_DIST_GN)

#                 if is_in_beam and is_available:
#                     chosen_gn[i] = g
#                     gn_taken[g] = True
#                     break

#         # --- 通信干扰 ---
#         p_main = real_actions[:, 2] * P_TX  # 通信+充电功率 (beta * P_TX)
#         p_side = (1 - real_actions[:, 2]) * P_TX  # 探测功率 ((1-beta) * P_TX)
#         rho = real_actions[:, 3]  # 通信比例

#         rate = np.zeros(NUM_UAV)  # 通信速率
#         eh = np.zeros(NUM_GN)  # 地面设备收集能量速率 (速率 W/s)


#         for i in range(NUM_UAV):
#             g = chosen_gn[i]
#             if g == -1:
#                 continue

#             g_main = avg_pathloss_linear(self.uav_pos[i], self.gn_pos[g])
#             interference = 0.0
#             for j in range(NUM_UAV):
#                 if j == i: continue
#                 g_int = avg_pathloss_linear(self.uav_pos[j], self.gn_pos[g])
#                 interference += p_main[j] * g_int * rho[j]

#             signal = rho[i] * p_main[i] * g_main
#             sinr = signal / (interference + ZIP0 * B + 1e-30)
#             rate[i] = B * math.log2(1 + sinr)

#             P_rx_eh = (1 - rho[i]) * p_main[i] * g_main
#             eh[g] += rf_energy_harvested(P_rx_eh) * 1e3  # 转换为 mJ/s

#         # GN 能量更新
#         pre_gn_energy = self.gn_energy.copy()
#         consumes = np.clip(np.random.normal(GN_BASE_CONSUME_MEAN, GN_BASE_CONSUME_STD, size=NUM_GN), 0.0, None) * DT
#         gains = eh * DT
#         self.gn_energy = np.clip(self.gn_energy - consumes + gains, 0.0, GN_E_MAX)
#         step_total_charge = np.sum(gains)

#         # 探测与干扰
#         radar_Pr = np.zeros((NUM_UAV, NUM_TARGET))
#         for i in range(NUM_UAV):
#             for n in range(NUM_TARGET):
#                 radar_Pr[i, n] = radar_received_power(self.uav_pos[i], self.target_pos[n], p_side[i])

#         valid_detection_matrix = np.zeros((NUM_UAV, NUM_TARGET), dtype=bool)
#         for i in range(NUM_UAV):
#             for n in range(NUM_TARGET):
#                 interference = 0.0
#                 for j in range(NUM_UAV):
#                     if j == i:
#                         continue
#                     inter = radar_Pr[j, n]
#                     interference += inter
#                 sinr_db = lin2db(radar_Pr[i, n] / (interference + N0 + 1e-30))

#                 if sinr_db >= MIN_SINR_DB and rate[i] >= MIN_RATE:
#                     valid_detection_matrix[i, n] = True

#         pre_target_detect_count = self.target_detect_count.copy()
#         for n in range(NUM_TARGET):
#             self.target_detect_count[n] += np.sum(valid_detection_matrix[:, n])

#         # UAV 能耗
#         uav_flight_energy_consumed = np.array([uav_flight_energy(v) for v in uav_speeds])
#         uav_comm_energy_consumed = P_TX * DT
#         uav_energy_consumed = uav_flight_energy_consumed + uav_comm_energy_consumed

#         # === 奖励计算 ===
#         sense_fair = jain_index(self.target_detect_count)
#         # charge_fair = jain_index(self.gn_energy)
#         charge_fair = jain_index(self.gn_energy - 13.3)
#         per_agent_rewards = np.zeros(NUM_UAV)

#         for i in range(NUM_UAV):
#             for n in range(NUM_TARGET):
#                 q = pre_target_detect_count[n]
#                 if valid_detection_matrix[i, n]:
#                     r_sense_incentive = W1 * (α ** q)
#                     per_agent_rewards[i] += r_sense_incentive

#         min_gn_energy_idx = np.argmin(pre_gn_energy)
#         for i in range(NUM_UAV):
#             if chosen_gn[i] == min_gn_energy_idx:
#                 per_agent_rewards[i] += W2

#         for i in range(NUM_UAV):
#             energy_i = uav_energy_consumed[i]
#             r_sense = W_SENSE * sense_fair * np.sum(self.target_detect_count)
#             r_charge = W_CHARGE * charge_fair * np.sum(self.gn_energy)
#             base = (r_sense + r_charge) / ((energy_i + 1e-6) * W_ENERGY)

#             penalty = 0.0
#             if collision_flags[i]:
#                 penalty += COLLISION_PENALTY
#             if oob_flags[i]:
#                 penalty += OUT_OF_BOUNDS_PENALTY

#             per_agent_rewards[i] += base - penalty

#         for g in range(NUM_GN):
#             if self.gn_energy[g] < GN_ENERGY_THRESHOLD:
#                 dists = np.linalg.norm(self.uav_pos - self.gn_pos[g], axis=1)
#                 nearest_uav = int(np.argmin(dists))
#                 per_agent_rewards[nearest_uav] -= W_PENALTY

#         #total_reward = np.mean(per_agent_rewards)

#         self.uav_prev_actions = actions.copy()
#         self.t += 1
#         done = (self.t >= self.episode_steps)

#         for i in range(NUM_UAV):
#             self.agent_paths[i].append(self.uav_pos[i].copy())

#         # --- 新增：存储当前步的探测结果 ---
#         self.detection_history.append(valid_detection_matrix.copy())

#         info = {
#             'uav_pos': self.uav_pos.copy(),
#             'chosen_gn': chosen_gn.copy(),
#             'gn_energy': self.gn_energy.copy(),
#             'target_detect_count': self.target_detect_count.copy(),
#             'collision_flags': collision_flags,
#             'oob_flags': oob_flags,
#             'valid_detection_matrix': valid_detection_matrix,
#             'sense_fair': sense_fair,
#             'charge_fair': charge_fair,
#             'step_uav_energy': np.mean(uav_energy_consumed),
#             'step_total_charge': step_total_charge
#         }
#         return self._get_obs(), per_agent_rewards, done, info

#     def render(self, save_dir=None, episode_index=None, plot_links=True):
#         """
#         渲染无人机的飞行轨迹，并在每步位置标记'x'，
#         同时绘制无人机到目标的“成功探测连线”。

#         Args:
#             save_dir (str, optional): 保存图像的目录。
#             episode_index (int, optional): 回合索引，用于文件名。
#             plot_links (bool): 是否绘制探测连线。
#         """
#         plt.figure(figsize=(10, 8))
#         ax = plt.gca()

#         # 绘制地面设备（蓝点）
#         plt.scatter(self.gn_pos[:, 0], self.gn_pos[:, 1], c='blue', label='Ground Nodes (GNs)', s=50, zorder=2)

#         # 绘制目标位置（红点）
#         plt.scatter(self.target_pos[:, 0], self.target_pos[:, 1], c='red', label='Targets', s=50, zorder=2)

#         # 绘制每个无人机的起始位置（大 "X"）
#         plt.scatter(UAV_START_POS[:, 0], UAV_START_POS[:, 1], c='green', marker='X', label='UAV Start Position', s=150,
#                     zorder=3)

#         colors = ['red', 'blue', 'green']
#         for i in range(NUM_UAV):
#             # 转换为numpy数组以便于操作
#             traj = np.array(self.agent_paths[i])

#             # 1. 绘制平滑的轨迹线
#             plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.7, label=f'UAV {i + 1} Trajectory',
#                      zorder=1)

#             # 2. 在每一步的位置绘制小'x'
#             # 从第二个点开始，因为第一个点是起始点
#             if len(traj) > 1:
#                 plt.scatter(traj[1:, 0], traj[1:, 1], marker='x', color=colors[i], s=20, alpha=0.8, zorder=3)

#         # 3. 绘制成功探测连线
#         link_plotted = False
#         if plot_links and len(self.detection_history) > 0:
#             for t in range(len(self.detection_history)):
#                 valid_matrix = self.detection_history[t]

#                 for i in range(NUM_UAV):
#                     # 获取UAV在t+1时刻的位置 (因为agent_paths[0]是初始位置)
#                     uav_pos = self.agent_paths[i][t + 1]

#                     for n in range(NUM_TARGET):
#                         if valid_matrix[i, n]:
#                             target_pos = self.target_pos[n]

#                             # 绘制从UAV到目标的虚线
#                             plt.plot([uav_pos[0], target_pos[0]],
#                                      [uav_pos[1], target_pos[1]],
#                                      color=colors[i],
#                                      linestyle=':',
#                                      linewidth=0.7,
#                                      alpha=0.4,
#                                      zorder=1)
#                             link_plotted = True

#         # 设置绘图区域范围
#         plt.xlim(-self.half_area, self.half_area)
#         plt.ylim(-self.half_area, self.half_area)
#         plt.gca().set_aspect('equal', adjustable='box')  # 保持x,y轴比例一致

#         # 标签、标题、图例
#         plt.xlabel("X position (m)")
#         plt.ylabel("Y position (m)")
#         plt.title(f"UAV Trajectories and Detections (Episode {episode_index})")

#         # --- 图例处理 ---
#         handles, labels = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))

#         if plot_links and link_plotted:
#             from matplotlib.lines import Line2D
#             link_legend = Line2D([0], [0], color='gray', linestyle=':',
#                                  linewidth=1, label='Successful Detection')
#             by_label[link_legend.get_label()] = link_legend

#         plt.legend(by_label.values(), by_label.keys(),
#                    loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
#         plt.grid(True)
#         plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整布局为图例留出空间

#         # 保存图像
#         if save_dir is not None and episode_index is not None:
#             os.makedirs(save_dir, exist_ok=True)
#             file_path = os.path.join(save_dir, f"episode_{episode_index}.png")
#             plt.savefig(file_path, bbox_inches='tight', dpi=150)
#             print(f"[Render] Saved trajectory plot: {file_path}")

#         # 可选：显示图像（非阻塞）
#         plt.show(block=False)
#         plt.pause(0.1)  # 暂停一下以确保图像有时间渲染
#         plt.close()

#     def close(self):
#         plt.close('all')  # 关闭所有matplotlib窗口

# # 充电奖励变为即时的
# import math
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import os
# import gym
# from gym.spaces import Box

# # ------------------- 常量参数定义（含单位说明） -------------------
# AREA_SIZE = 500.0  # m，环境边长
# HALF_AREA = AREA_SIZE / 2.0  # m, 半边长
# DT = 1.0  # s，每时间步长度

# NUM_UAV = 3  # 无人机数量
# NUM_TARGET = 20  # 目标数量
# NUM_GN = 10  # 地面设备数量

# # 无人机起始位置（m），从 [0, 500] 移到 [-250, 250]
# UAV_START_POS_ORIGINAL = np.array([[50.0, 50.0], [350.0, 100.0], [200.0, 400.0]])
# UAV_START_POS = UAV_START_POS_ORIGINAL - HALF_AREA

# # 高度参数
# H_UAV = 15.0  # m，无人机高度
# H_GN = 5.0  # m，地面设备高度
# H_TARGET = 0.0  # m，目标高度

# # 运动限制
# D_MAX = 40.0  # m，每步最大移动距离 (最大欧氏距离)
# SAFE_DISTANCE = 40.0  # m，无人机最小安全距离

# # 通信与感知参数
# P_TX = 30.0  # W，无人机发射功率
# B = 2e6  # Hz，通信带宽
# FC = 3e9  # Hz，载波频率
# C = 3e8  # m/s，光速
# LAMBDA = C / FC  # m，波长
# G_TX = 10 ** 1.3  # 无量纲，发射天线增益
# G_RX = 10 ** 1.3  # 无量纲，接收天线增益
# SIGMA_RCS = 0.25  # m²，雷达散射截面

# KB = 1.38e-23  # J/K，玻尔兹曼常数
# TEMP_K = 290.0  # K，噪声温度
# NOISE_FIG_DB = 10 ** 0.5  # dB，噪声系数
# N0 = KB * TEMP_K * B * NOISE_FIG_DB  # 噪声功率

# # 信道参数
# LOS_A = 4.88
# LOS_B = 0.43
# ATT_LOS = 0.1  # LoS 衰减系数
# ATT_NLOS = 21  # NLoS 衰减系数
# ZIP0 = 10 ** -20.4  # 噪声dBm/Hz

# # 能量收集参数（文档公式）
# EH_U = 24e-3
# EH_r1 = 150
# EH_r2 = 0.014

# # GN 能量模型参数
# GN_E_MAX = 50.0  # mJ，地面设备最大能量
# GN_E_INIT = 20.0  # mJ，初始能量
# GN_BASE_CONSUME_MEAN = 0.08  # mJ/s，平均耗能速率
# GN_BASE_CONSUME_STD = 0.02  # mJ/s，耗能标准差

# # UAV 飞行能耗模型参数
# V_TIP = 120.0  # m/s，旋翼叶尖速度
# C1 = 80.0  # W，桨叶剖面功率系数
# C2 = 22.0
# C3 = 263.4  # W，诱导功率系数
# C4 = 0.0092  # 无量纲，寄生功率系数

# # 奖励权重与惩罚
# W_SENSE = 5.0
# W_CHARGE = 2000
# W_ENERGY = 0.5
# W_PENALTY = 50.0
# COLLISION_PENALTY = 20.0
# OUT_OF_BOUNDS_PENALTY = 20.0
# W1 = 1000  # 首次探测的奖励
# α = 0.1  # 衰减因子
# W2 = 300  # 充电激励奖励
# EXPLORATION_REWARD = 10.0  # 连接失败奖励

# GN_ENERGY_THRESHOLD = 13.5  # mJ
# MIN_SINR_DB = 16.0  # dB，探测 SINR 阈值
# MIN_RATE = 1e5  # bps，通信速率阈值
# THETA_HALF_BW_DEG_GN = 60.0  # 通信范围

# # 随机种子
# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)


# # ------------------- 工具函数 -------------------
# # 动作空间 a \in [-1, 1]
# def angle_from_norm(theta_norm):
#     """角度：[-1, 1] -> [-pi, pi]"""
#     return float(theta_norm) * math.pi


# def dist_from_norm(dist_norm):
#     """距离：[-1, 1] -> [-D_MAX, D_MAX]"""
#     return float(dist_norm) * D_MAX


# def beta_from_norm(beta_norm):
#     """beta: [-1, 1] -> [0, 1]"""
#     return (float(beta_norm) + 1.0) / 2.0


# def rho_from_norm(rho_norm):
#     """rho: [-1, 1] -> [0, 1]"""
#     return (float(rho_norm) + 1.0) / 2.0


# def los_probability(uav_pos, gn_pos):
#     """LoS 概率模型（文档公式）"""
#     dx = gn_pos[0] - uav_pos[0]
#     dy = gn_pos[1] - uav_pos[1]
#     horiz = math.hypot(dx, dy)
#     elev = math.atan2(H_UAV - H_GN, horiz)
#     elev_deg = math.degrees(elev)
#     p = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (elev_deg - LOS_A)))
#     return p


# def avg_pathloss_linear(uav_pos, gn_pos, theta_half_bw_deg=60.0):
#     """平均信道功率增益（含定向天线影响）"""
#     dx = gn_pos[0] - uav_pos[0]
#     dy = gn_pos[1] - uav_pos[1]
#     d3d = math.hypot(math.hypot(dx, dy), H_UAV - H_GN)
#     p_l = los_probability(uav_pos, gn_pos)
#     eta_avg = (d3d ** -2) * ((4 * math.pi * FC / C) ** -2) * ((p_l * ATT_LOS + (1 - p_l) * ATT_NLOS) ** -1)
#     theta_rad = math.radians(theta_half_bw_deg)
#     horiz = math.hypot(dx, dy)
#     if horiz <= (H_UAV - H_GN) * math.tan(theta_rad):
#         dir_gain_factor = 2.28 / (theta_rad ** 2)
#         h = eta_avg * dir_gain_factor
#     else:
#         h = eta_avg
#     return h


# def radar_received_power(uav_pos, target_pos, p_side):
#     """雷达回波功率（文档公式）"""
#     dx = target_pos[0] - uav_pos[0]
#     dy = target_pos[1] - uav_pos[1]
#     R = math.hypot(math.hypot(dx, dy), H_UAV - H_TARGET)
#     Pr = p_side * G_TX * G_RX * (LAMBDA ** 2 * SIGMA_RCS) / ((4.0 * math.pi) ** 3 * (R ** 4))
#     return Pr


# def comm_rx_power(uav_pos, gn_pos, p_main, rho):
#     """通信接收功率 = ρ * p_main * g"""
#     g = avg_pathloss_linear(uav_pos, gn_pos)
#     P_rx = rho * p_main * g
#     return float(max(P_rx, 0.0))


# def rf_energy_harvested(p_rx):
#     """能量收集电路模型（文档公式）"""
#     val = EH_U / (1 + math.exp(EH_r1 * EH_r2)) * ((1 + math.exp(EH_r1 * EH_r2)) /
#                                                   (1 + math.exp(EH_r1 * EH_r2) * math.exp(-EH_r1 * p_rx)) - 1)
#     return val


# def uav_flight_energy(v):
#     """无人机能耗（文档物理模型）"""
#     # 注意：v 是速度的绝对值
#     term1 = C1 * (1.0 + 3.0 * (v ** 2) / (V_TIP ** 2))
#     term2 = C2 * math.sqrt(math.sqrt(C3 + (v ** 4) / 4) - (v ** 2) / 2)
#     term3 = C4 * (v ** 3)
#     P_total = term1 + term2 + term3
#     return P_total * DT


# def lin2db(x):
#     return 10.0 * math.log10(max(x, 1e-30))


# def jain_index(x):
#     x = np.array(x, dtype=float)
#     num = (np.sum(x) ** 2)
#     den = len(x) * np.sum(x ** 2) + 1e-12
#     return float(num / den)


# # ------------------- 环境定义 -------------------
# class UAVEnvFixed:
#     def __init__(self,
#                  target_pos_list=None,
#                  gn_pos_list=None,
#                  gn_energy_init=GN_E_INIT,
#                  episode_steps=80):
#         self.area = AREA_SIZE
#         self.half_area = HALF_AREA  # [-250, 250]
#         self.episode_steps = episode_steps

#         # 原始固定位置 (在 [0, 500])
#         target_pos_original = np.array([
#             [50.0, 120.0], [200.0, 180.0], [300.0, 60.0], [400.0, 350.0], [120.0, 450.0],
#             [250.0, 400.0], [60.0, 320.0], [470.0, 200.0], [100.0, 280.0], [370.0, 450.0],
#             [50.0, 100.0], [420.0, 150.0], [150.0, 240.0], [320.0, 50.0], [380.0, 120.0],
#             [100.0, 400.0], [270.0, 150.0], [220.0, 220.0], [60.0, 250.0], [420.0, 70.0]
#         ])

#         gn_pos_original = np.array([
#             [30.0, 220.0], [160.0, 180.0], [240.0, 30.0], [370.0, 400.0], [50.0, 90.0],
#             [310.0, 330.0], [320.0, 180.0], [440.0, 270.0], [180.0, 440.0], [90.0, 400.0]
#         ])

#         # 转换为 [-250, 250] 坐标系
#         self.target_pos = (target_pos_original - self.half_area) if target_pos_list is None else (
#                 np.array(target_pos_list) - self.half_area)
#         self.gn_pos = (gn_pos_original - self.half_area) if gn_pos_list is None else (
#                 np.array(gn_pos_list) - self.half_area)

#         # 状态初始化
#         self.uav_pos = UAV_START_POS.copy()
#         self.uav_prev_pos = self.uav_pos.copy()
#         self.uav_prev_actions = np.zeros((NUM_UAV, 4))  # 存储的是归一化后的动作 [-1, 1]
#         self.gn_energy = np.ones(NUM_GN) * gn_energy_init
#         self.target_detect_count = np.zeros(NUM_TARGET, dtype=int)
#         self.t = 0
#         self.agent_paths = [[] for _ in range(NUM_UAV)]  # 用于存储每个无人机的轨迹

#         # --- 新增：为render函数存储历史探测数据 ---
#         self.detection_history = []

#         # --- 为 MARL 框架添加的属性 ---
#         self.agent_num = NUM_UAV
#         self.uav_num = NUM_UAV  # 兼容 PPO 和 MAAS

#         # 观测维度 = 6 (xn, yn, prev_dist, prev_theta, prev_beta, prev_rho)
#         self.observation_space_len = 6
#         # 动作维度 = 4 (theta, dist, beta, rho)
#         self.action_space_len = 4

#         # 定义 Gym 空间 (PPO 和 DDPG/SAC 的 agents_config_dict 需要)
#         # 观测空间 [-1, 1]，注意形状是 (obs_len,)
#         self.observation_space = Box(-1.0, 1.0, (self.observation_space_len,))
#         # 动作空间 [-1, 1]，注意形状是 (act_len,)
#         self.action_space = Box(-1.0, 1.0, (self.action_space_len,))
#         # --- MARL 属性添加完毕 ---

#         # 轨迹初始化 (UAV_START_POS 已经在 [-250, 250] 范围内)
#         for i in range(NUM_UAV):
#             self.agent_paths[i].append(self.uav_pos[i].copy())

#     def reset(self):
#         self.uav_pos = UAV_START_POS.copy()
#         self.uav_prev_pos = self.uav_pos.copy()
#         self.uav_prev_actions.fill(0.0)
#         self.gn_energy = np.ones(NUM_GN) * GN_E_INIT
#         self.target_detect_count[:] = 0
#         self.t = 0

#         # --- 新增：重置历史记录 ---
#         self.detection_history = []

#         # 初始化 UAV 的轨迹
#         self.agent_paths = [[] for _ in range(NUM_UAV)]
#         for i in range(NUM_UAV):
#             self.agent_paths[i].append(self.uav_pos[i].copy())

#         return self._get_obs()

#     def _get_obs(self):
#         obs = []
#         for i in range(NUM_UAV):
#             x, y = self.uav_pos[i]
#             # 归一化到 [-1, 1] 范围
#             xn, yn = x / self.half_area, y / self.half_area
#             prev = self.uav_prev_actions[i]
#             # prev[0]=theta, prev[1]=dist, prev[2]=beta, prev[3]=rho (都是归一化的)
#             s = np.array([xn, yn, prev[1], prev[0], prev[2], prev[3]], dtype=np.float32)
#             obs.append(s)
#         return np.stack(obs, axis=0)

#     def step(self, actions):
#         # 动作 a \in [-1, 1]
#         actions = np.clip(np.array(actions, dtype=float), -1.0, 1.0)

#         # 实际动作 (real_actions) 存储转换后的物理值：[theta_rad, d_m, beta_[0,1], rho_[0,1]]
#         real_actions = np.zeros_like(actions)
#         for i in range(NUM_UAV):
#             theta = angle_from_norm(actions[i, 0])  # [-pi, pi]
#             d = dist_from_norm(actions[i, 1])  # [-D_MAX, D_MAX]
#             beta = beta_from_norm(actions[i, 2])  # [0, 1]
#             rho = rho_from_norm(actions[i, 3])  # [0, 1]
#             real_actions[i] = [theta, d, beta, rho]

#         # UAV 移动
#         self.uav_prev_pos = self.uav_pos.copy()
#         new_positions = self.uav_pos.copy()
#         uav_speeds = np.zeros(NUM_UAV)
#         for i in range(NUM_UAV):
#             theta, d = real_actions[i, 0], real_actions[i, 1]
#             dx = d * math.cos(theta)
#             dy = d * math.sin(theta)
#             nx = self.uav_pos[i, 0] + dx
#             ny = self.uav_pos[i, 1] + dy
#             new_positions[i] = [nx, ny]
#             # 速度 v = |d| / DT
#             uav_speeds[i] = abs(d) / DT
#         self.uav_pos = new_positions.copy()

#         # 违规检测
#         collision_flags = np.zeros(NUM_UAV, dtype=int)  # 碰撞
#         oob_flags = np.zeros(NUM_UAV, dtype=int)  # 越界

#         lower_bound = -self.half_area
#         upper_bound = self.half_area

#         for i in range(NUM_UAV):
#             x, y = self.uav_pos[i]
#             # 越界检测: 范围 [-250, 250]
#             if x < lower_bound or x > upper_bound or y < lower_bound or y > upper_bound:
#                 oob_flags[i] = 1
#                 self.uav_pos[i] = self.uav_prev_pos[i].copy()

#         for i in range(NUM_UAV):
#             for j in range(i + 1, NUM_UAV):
#                 di = np.linalg.norm(self.uav_pos[i] - self.uav_pos[j])
#                 if di < SAFE_DISTANCE:
#                     collision_flags[i] = collision_flags[j] = 1
#                     # 避免在同一时间步内多次回退
#                     if oob_flags[i] == 0: self.uav_pos[i] = self.uav_prev_pos[i].copy()
#                     if oob_flags[j] == 0: self.uav_pos[j] = self.uav_prev_pos[j].copy()

#         # GN 分配（基于定向天线约束）
#         chosen_gn = -np.ones(NUM_UAV, dtype=int)
#         gn_taken = np.zeros(NUM_GN, dtype=bool)

#         THETA_HALF_BW_RAD_GN = math.radians(THETA_HALF_BW_DEG_GN)
#         MAX_BEAM_HORIZ_DIST_GN = (H_UAV - H_GN) * math.tan(THETA_HALF_BW_RAD_GN)

#         # if self.t == 0:
#         #     print(f"[Env Info] Max horizontal distance for GN beam: {MAX_BEAM_HORIZ_DIST_GN:.2f} m")

#         for i in range(NUM_UAV):
#             dists = np.linalg.norm(self.gn_pos - self.uav_pos[i], axis=1)
#             order = np.argsort(dists)

#             for g in order:
#                 is_available = not gn_taken[g]
#                 is_in_beam = (dists[g] <= MAX_BEAM_HORIZ_DIST_GN)

#                 if is_in_beam and is_available:
#                     chosen_gn[i] = g
#                     gn_taken[g] = True
#                     break

#         # --- 通信干扰 ---
#         p_main = real_actions[:, 2] * P_TX  # 通信+充电功率 (beta * P_TX)
#         p_side = (1 - real_actions[:, 2]) * P_TX  # 探测功率 ((1-beta) * P_TX)
#         rho = real_actions[:, 3]  # 通信比例

#         rate = np.zeros(NUM_UAV)  # 通信速率
#         eh = np.zeros(NUM_GN)  # 地面设备收集能量速率 (速率 W/s)

#         # --- 新增: 记录每个UAV贡献的充电量 (单位: mJ) ---
#         uav_charge_gains = np.zeros(NUM_UAV)
#         # --- 新增结束 ---


#         for i in range(NUM_UAV):
#             g = chosen_gn[i]
#             if g == -1:
#                 continue

#             g_main = avg_pathloss_linear(self.uav_pos[i], self.gn_pos[g])
#             interference = 0.0
#             for j in range(NUM_UAV):
#                 if j == i: continue
#                 g_int = avg_pathloss_linear(self.uav_pos[j], self.gn_pos[g])
#                 interference += p_main[j] * g_int * rho[j]

#             signal = rho[i] * p_main[i] * g_main
#             sinr = signal / (interference + ZIP0 * B + 1e-30)
#             rate[i] = B * math.log2(1 + sinr)

#             P_rx_eh = (1 - rho[i]) * p_main[i] * g_main
#             eh[g] += rf_energy_harvested(P_rx_eh) * 1e3  # 转换为 mJ/s
#             # 3. 记录该UAV在本时隙贡献的 *总能量* (mJ)
#             uav_charge_gains[i] = rf_energy_harvested(P_rx_eh) * 1e3 * DT

#         # GN 能量更新
#         pre_gn_energy = self.gn_energy.copy()
#         consumes = np.clip(np.random.normal(GN_BASE_CONSUME_MEAN, GN_BASE_CONSUME_STD, size=NUM_GN), 0.0, None) * DT
#         gains = eh * DT
#         self.gn_energy = np.clip(self.gn_energy - consumes + gains, 0.0, GN_E_MAX)
#         step_total_charge = np.sum(gains)

#         # 探测与干扰
#         radar_Pr = np.zeros((NUM_UAV, NUM_TARGET))
#         for i in range(NUM_UAV):
#             for n in range(NUM_TARGET):
#                 radar_Pr[i, n] = radar_received_power(self.uav_pos[i], self.target_pos[n], p_side[i])

#         valid_detection_matrix = np.zeros((NUM_UAV, NUM_TARGET), dtype=bool)
#         for i in range(NUM_UAV):
#             for n in range(NUM_TARGET):
#                 interference = 0.0
#                 for j in range(NUM_UAV):
#                     if j == i:
#                         continue
#                     inter = radar_Pr[j, n]
#                     interference += inter
#                 sinr_db = lin2db(radar_Pr[i, n] / (interference + N0 + 1e-30))

#                 if sinr_db >= MIN_SINR_DB and rate[i] >= MIN_RATE:
#                     valid_detection_matrix[i, n] = True

#         pre_target_detect_count = self.target_detect_count.copy()
#         for n in range(NUM_TARGET):
#             self.target_detect_count[n] += np.sum(valid_detection_matrix[:, n])

#         # UAV 能耗
#         uav_flight_energy_consumed = np.array([uav_flight_energy(v) for v in uav_speeds])
#         uav_comm_energy_consumed = P_TX * DT
#         uav_energy_consumed = uav_flight_energy_consumed + uav_comm_energy_consumed

#         # === 奖励计算 ===
#         sense_fair = jain_index(self.target_detect_count)
#         # charge_fair = jain_index(self.gn_energy)
#         charge_fair = jain_index(self.gn_energy - 13.3)
#         per_agent_rewards = np.zeros(NUM_UAV)

#         for i in range(NUM_UAV):
#             for n in range(NUM_TARGET):
#                 q = pre_target_detect_count[n]
#                 if valid_detection_matrix[i, n]:
#                     r_sense_incentive = W1 * (α ** q)
#                     per_agent_rewards[i] += r_sense_incentive

#         min_gn_energy_idx = np.argmin(pre_gn_energy)
#         for i in range(NUM_UAV):
#             if chosen_gn[i] == min_gn_energy_idx:
#                 per_agent_rewards[i] += W2

#         for i in range(NUM_UAV):
#             energy_i = uav_energy_consumed[i]
#             r_sense = W_SENSE * sense_fair * np.sum(self.target_detect_count)
#             r_charge = W_CHARGE * charge_fair * uav_charge_gains[i]
#             base = (r_sense + r_charge) / ((energy_i + 1e-6) * W_ENERGY)

#             penalty = 0.0
#             if collision_flags[i]:
#                 penalty += COLLISION_PENALTY
#             if oob_flags[i]:
#                 penalty += OUT_OF_BOUNDS_PENALTY

#             per_agent_rewards[i] += base - penalty
#             if chosen_gn[i] == -1:
#                 per_agent_rewards[i] += EXPLORATION_REWARD

#         for g in range(NUM_GN):
#             if self.gn_energy[g] < GN_ENERGY_THRESHOLD:
#                 dists = np.linalg.norm(self.uav_pos - self.gn_pos[g], axis=1)
#                 nearest_uav = int(np.argmin(dists))
#                 per_agent_rewards[nearest_uav] -= W_PENALTY

#         #total_reward = np.mean(per_agent_rewards)

#         self.uav_prev_actions = actions.copy()
#         self.t += 1
#         done = (self.t >= self.episode_steps)

#         for i in range(NUM_UAV):
#             self.agent_paths[i].append(self.uav_pos[i].copy())

#         # --- 新增：存储当前步的探测结果 ---
#         self.detection_history.append(valid_detection_matrix.copy())

#         info = {
#             'uav_pos': self.uav_pos.copy(),
#             'chosen_gn': chosen_gn.copy(),
#             'gn_energy': self.gn_energy.copy(),
#             'target_detect_count': self.target_detect_count.copy(),
#             'collision_flags': collision_flags,
#             'oob_flags': oob_flags,
#             'valid_detection_matrix': valid_detection_matrix,
#             'sense_fair': sense_fair,
#             'charge_fair': charge_fair,
#             'step_uav_energy': np.mean(uav_energy_consumed),
#             'step_total_charge': step_total_charge
#         }
#         return self._get_obs(), per_agent_rewards, done, info

#     def render(self, save_dir=None, episode_index=None, plot_links=True):
#         """
#         渲染无人机的飞行轨迹，并在每步位置标记'x'，
#         同时绘制无人机到目标的“成功探测连线”。

#         Args:
#             save_dir (str, optional): 保存图像的目录。
#             episode_index (int, optional): 回合索引，用于文件名。
#             plot_links (bool): 是否绘制探测连线。
#         """
#         plt.figure(figsize=(10, 8))
#         ax = plt.gca()

#         # 绘制地面设备（蓝点）
#         plt.scatter(self.gn_pos[:, 0], self.gn_pos[:, 1], c='blue', label='Ground Nodes (GNs)', s=50, zorder=2)

#         # 绘制目标位置（红点）
#         plt.scatter(self.target_pos[:, 0], self.target_pos[:, 1], c='red', label='Targets', s=50, zorder=2)

#         # 绘制每个无人机的起始位置（大 "X"）
#         plt.scatter(UAV_START_POS[:, 0], UAV_START_POS[:, 1], c='green', marker='X', label='UAV Start Position', s=150,
#                     zorder=3)

#         colors = ['red', 'blue', 'green']
#         for i in range(NUM_UAV):
#             # 转换为numpy数组以便于操作
#             traj = np.array(self.agent_paths[i])

#             # 1. 绘制平滑的轨迹线
#             plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.7, label=f'UAV {i + 1} Trajectory',
#                      zorder=1)

#             # 2. 在每一步的位置绘制小'x'
#             # 从第二个点开始，因为第一个点是起始点
#             if len(traj) > 1:
#                 plt.scatter(traj[1:, 0], traj[1:, 1], marker='x', color=colors[i], s=20, alpha=0.8, zorder=3)

#         # 3. 绘制成功探测连线
#         link_plotted = False
#         if plot_links and len(self.detection_history) > 0:
#             for t in range(len(self.detection_history)):
#                 valid_matrix = self.detection_history[t]

#                 for i in range(NUM_UAV):
#                     # 获取UAV在t+1时刻的位置 (因为agent_paths[0]是初始位置)
#                     uav_pos = self.agent_paths[i][t + 1]

#                     for n in range(NUM_TARGET):
#                         if valid_matrix[i, n]:
#                             target_pos = self.target_pos[n]

#                             # 绘制从UAV到目标的虚线
#                             plt.plot([uav_pos[0], target_pos[0]],
#                                      [uav_pos[1], target_pos[1]],
#                                      color=colors[i],
#                                      linestyle=':',
#                                      linewidth=0.7,
#                                      alpha=0.4,
#                                      zorder=1)
#                             link_plotted = True

#         # 设置绘图区域范围
#         plt.xlim(-self.half_area, self.half_area)
#         plt.ylim(-self.half_area, self.half_area)
#         plt.gca().set_aspect('equal', adjustable='box')  # 保持x,y轴比例一致

#         # 标签、标题、图例
#         plt.xlabel("X position (m)")
#         plt.ylabel("Y position (m)")
#         plt.title(f"UAV Trajectories and Detections (Episode {episode_index})")

#         # --- 图例处理 ---
#         handles, labels = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))

#         if plot_links and link_plotted:
#             from matplotlib.lines import Line2D
#             link_legend = Line2D([0], [0], color='gray', linestyle=':',
#                                  linewidth=1, label='Successful Detection')
#             by_label[link_legend.get_label()] = link_legend

#         plt.legend(by_label.values(), by_label.keys(),
#                    loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
#         plt.grid(True)
#         plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整布局为图例留出空间

#         # 保存图像
#         if save_dir is not None and episode_index is not None:
#             os.makedirs(save_dir, exist_ok=True)
#             file_path = os.path.join(save_dir, f"episode_{episode_index}.png")
#             plt.savefig(file_path, bbox_inches='tight', dpi=150)
#             print(f"[Render] Saved trajectory plot: {file_path}")

#         # 可选：显示图像（非阻塞）
#         plt.show(block=False)
#         plt.pause(0.1)  # 暂停一下以确保图像有时间渲染
#         plt.close()

#     def close(self):
#         plt.close('all')  # 关闭所有matplotlib窗口

# # 探测奖励也变成即时的
# import math
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import os
# import gym
# from gym.spaces import Box

# # ------------------- 常量参数定义（含单位说明） -------------------
# AREA_SIZE = 500.0  # m，环境边长
# HALF_AREA = AREA_SIZE / 2.0  # m, 半边长
# DT = 1.0  # s，每时间步长度

# NUM_UAV = 3  # 无人机数量
# NUM_TARGET = 20  # 目标数量
# NUM_GN = 10  # 地面设备数量

# # 无人机起始位置（m），从 [0, 500] 移到 [-250, 250]
# UAV_START_POS_ORIGINAL = np.array([[50.0, 50.0], [350.0, 100.0], [200.0, 400.0]])
# UAV_START_POS = UAV_START_POS_ORIGINAL - HALF_AREA

# # 高度参数
# H_UAV = 15.0  # m，无人机高度
# H_GN = 5.0  # m，地面设备高度
# H_TARGET = 0.0  # m，目标高度

# # 运动限制
# D_MAX = 40.0  # m，每步最大移动距离 (最大欧氏距离)
# SAFE_DISTANCE = 40.0  # m，无人机最小安全距离

# # 通信与感知参数
# P_TX = 30.0  # W，无人机发射功率
# B = 2e6  # Hz，通信带宽
# FC = 3e9  # Hz，载波频率
# C = 3e8  # m/s，光速
# LAMBDA = C / FC  # m，波长
# G_TX = 10 ** 1.3  # 无量纲，发射天线增益
# G_RX = 10 ** 1.3  # 无量纲，接收天线增益
# SIGMA_RCS = 0.25  # m²，雷达散射截面

# KB = 1.38e-23  # J/K，玻尔兹曼常数
# TEMP_K = 290.0  # K，噪声温度
# NOISE_FIG_DB = 10 ** 0.5  # dB，噪声系数
# N0 = KB * TEMP_K * B * NOISE_FIG_DB  # 噪声功率

# # 信道参数
# LOS_A = 4.88
# LOS_B = 0.43
# ATT_LOS = 0.1  # LoS 衰减系数
# ATT_NLOS = 21  # NLoS 衰减系数
# ZIP0 = 10 ** -20.4  # 噪声dBm/Hz

# # 能量收集参数（文档公式）
# EH_U = 24e-3
# EH_r1 = 150
# EH_r2 = 0.014

# # GN 能量模型参数
# GN_E_MAX = 50.0  # mJ，地面设备最大能量
# GN_E_INIT = 20.0  # mJ，初始能量
# GN_BASE_CONSUME_MEAN = 0.08  # mJ/s，平均耗能速率
# GN_BASE_CONSUME_STD = 0.02  # mJ/s，耗能标准差

# # UAV 飞行能耗模型参数
# V_TIP = 120.0  # m/s，旋翼叶尖速度
# C1 = 80.0  # W，桨叶剖面功率系数
# C2 = 22.0
# C3 = 263.4  # W，诱导功率系数
# C4 = 0.0092  # 无量纲，寄生功率系数

# # 奖励权重与惩罚
# W_SENSE = 10.0
# W_CHARGE = 5000.0
# W_ENERGY = 0.6
# W_PENALTY = 50.0
# COLLISION_PENALTY = 20.0
# OUT_OF_BOUNDS_PENALTY = 20.0
# W1 = 40  # 首次探测的奖励
# α = 0.2  # 衰减因子
# W2 = 20  # 充电激励奖励
# EXPLORATION_REWARD = 5.0  # 连接失败奖励

# GN_ENERGY_THRESHOLD = 13.5  # mJ
# MIN_SINR_DB = 15.0  # dB，探测 SINR 阈值
# MIN_RATE = 1e5  # bps，通信速率阈值
# THETA_HALF_BW_DEG_GN = 60.0  # 通信范围

# # 随机种子
# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)


# # ------------------- 工具函数 -------------------
# # 动作空间 a \in [-1, 1]
# def angle_from_norm(theta_norm):
#     """角度：[-1, 1] -> [-pi, pi]"""
#     return float(theta_norm) * math.pi


# def dist_from_norm(dist_norm):
#     """距离：[-1, 1] -> [-D_MAX, D_MAX]"""
#     return float(dist_norm) * D_MAX


# def beta_from_norm(beta_norm):
#     """beta: [-1, 1] -> [0, 1]"""
#     return (float(beta_norm) + 1.0) / 2.0


# def rho_from_norm(rho_norm):
#     """rho: [-1, 1] -> [0, 1]"""
#     return (float(rho_norm) + 1.0) / 2.0


# def los_probability(uav_pos, gn_pos):
#     """LoS 概率模型（文档公式）"""
#     dx = gn_pos[0] - uav_pos[0]
#     dy = gn_pos[1] - uav_pos[1]
#     horiz = math.hypot(dx, dy)
#     elev = math.atan2(H_UAV - H_GN, horiz)
#     elev_deg = math.degrees(elev)
#     p = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (elev_deg - LOS_A)))
#     return p


# def avg_pathloss_linear(uav_pos, gn_pos, theta_half_bw_deg=60.0):
#     """平均信道功率增益（含定向天线影响）"""
#     dx = gn_pos[0] - uav_pos[0]
#     dy = gn_pos[1] - uav_pos[1]
#     d3d = math.hypot(math.hypot(dx, dy), H_UAV - H_GN)
#     p_l = los_probability(uav_pos, gn_pos)
#     eta_avg = (d3d ** -2) * ((4 * math.pi * FC / C) ** -2) * ((p_l * ATT_LOS + (1 - p_l) * ATT_NLOS) ** -1)
#     theta_rad = math.radians(theta_half_bw_deg)
#     horiz = math.hypot(dx, dy)
#     if horiz <= (H_UAV - H_GN) * math.tan(theta_rad):
#         dir_gain_factor = 2.28 / (theta_rad ** 2)
#         h = eta_avg * dir_gain_factor
#     else:
#         h = eta_avg
#     return h


# def radar_received_power(uav_pos, target_pos, p_side):
#     """雷达回波功率（文档公式）"""
#     dx = target_pos[0] - uav_pos[0]
#     dy = target_pos[1] - uav_pos[1]
#     R = math.hypot(math.hypot(dx, dy), H_UAV - H_TARGET)
#     Pr = p_side * G_TX * G_RX * (LAMBDA ** 2 * SIGMA_RCS) / ((4.0 * math.pi) ** 3 * (R ** 4))
#     return Pr


# def comm_rx_power(uav_pos, gn_pos, p_main, rho):
#     """通信接收功率 = ρ * p_main * g"""
#     g = avg_pathloss_linear(uav_pos, gn_pos)
#     P_rx = rho * p_main * g
#     return float(max(P_rx, 0.0))


# def rf_energy_harvested(p_rx):
#     """能量收集电路模型（文档公式）"""
#     val = EH_U / (1 + math.exp(EH_r1 * EH_r2)) * ((1 + math.exp(EH_r1 * EH_r2)) /
#                                                   (1 + math.exp(EH_r1 * EH_r2) * math.exp(-EH_r1 * p_rx)) - 1)
#     return val


# def uav_flight_energy(v):
#     """无人机能耗（文档物理模型）"""
#     # 注意：v 是速度的绝对值
#     term1 = C1 * (1.0 + 3.0 * (v ** 2) / (V_TIP ** 2))
#     term2 = C2 * math.sqrt(math.sqrt(C3 + (v ** 4) / 4) - (v ** 2) / 2)
#     term3 = C4 * (v ** 3)
#     P_total = term1 + term2 + term3
#     return P_total * DT


# def lin2db(x):
#     return 10.0 * math.log10(max(x, 1e-30))


# def jain_index(x):
#     x = np.array(x, dtype=float)
#     num = (np.sum(x) ** 2)
#     den = len(x) * np.sum(x ** 2) + 1e-12
#     return float(num / den)


# # ------------------- 环境定义 -------------------
# class UAVEnvFixed:
#     def __init__(self,
#                  target_pos_list=None,
#                  gn_pos_list=None,
#                  gn_energy_init=GN_E_INIT,
#                  episode_steps=80):
#         self.area = AREA_SIZE
#         self.half_area = HALF_AREA  # [-250, 250]
#         self.episode_steps = episode_steps

#         # 原始固定位置 (在 [0, 500])
#         target_pos_original = np.array([
#             [50.0, 120.0], [200.0, 180.0], [300.0, 60.0], [400.0, 350.0], [120.0, 450.0],
#             [250.0, 400.0], [60.0, 320.0], [470.0, 200.0], [100.0, 280.0], [370.0, 450.0],
#             [50.0, 100.0], [420.0, 150.0], [150.0, 240.0], [320.0, 50.0], [380.0, 120.0],
#             [100.0, 400.0], [270.0, 150.0], [220.0, 220.0], [60.0, 250.0], [420.0, 70.0]
#         ])

#         gn_pos_original = np.array([
#             [30.0, 220.0], [160.0, 180.0], [240.0, 30.0], [370.0, 400.0], [50.0, 90.0],
#             [310.0, 330.0], [320.0, 180.0], [440.0, 270.0], [180.0, 440.0], [90.0, 400.0]
#         ])

#         # 转换为 [-250, 250] 坐标系
#         self.target_pos = (target_pos_original - self.half_area) if target_pos_list is None else (
#                 np.array(target_pos_list) - self.half_area)
#         self.gn_pos = (gn_pos_original - self.half_area) if gn_pos_list is None else (
#                 np.array(gn_pos_list) - self.half_area)

#         # 状态初始化
#         self.uav_pos = UAV_START_POS.copy()
#         self.uav_prev_pos = self.uav_pos.copy()
#         self.uav_prev_actions = np.zeros((NUM_UAV, 4))  # 存储的是归一化后的动作 [-1, 1]
#         self.gn_energy = np.ones(NUM_GN) * gn_energy_init
#         self.target_detect_count = np.zeros(NUM_TARGET, dtype=int)
#         self.t = 0
#         self.agent_paths = [[] for _ in range(NUM_UAV)]  # 用于存储每个无人机的轨迹

#         # --- 新增：为render函数存储历史探测数据 ---
#         self.detection_history = []

#         # --- 为 MARL 框架添加的属性 ---
#         self.agent_num = NUM_UAV
#         self.uav_num = NUM_UAV  # 兼容 PPO 和 MAAS

#         # 观测维度 = 6 (xn, yn, prev_dist, prev_theta, prev_beta, prev_rho)
#         self.observation_space_len = 6
#         # 动作维度 = 4 (theta, dist, beta, rho)
#         self.action_space_len = 4

#         # 定义 Gym 空间 (PPO 和 DDPG/SAC 的 agents_config_dict 需要)
#         # 观测空间 [-1, 1]，注意形状是 (obs_len,)
#         self.observation_space = Box(-1.0, 1.0, (self.observation_space_len,))
#         # 动作空间 [-1, 1]，注意形状是 (act_len,)
#         self.action_space = Box(-1.0, 1.0, (self.action_space_len,))
#         # --- MARL 属性添加完毕 ---

#         # 轨迹初始化 (UAV_START_POS 已经在 [-250, 250] 范围内)
#         for i in range(NUM_UAV):
#             self.agent_paths[i].append(self.uav_pos[i].copy())

#     def reset(self):
#         self.uav_pos = UAV_START_POS.copy()
#         self.uav_prev_pos = self.uav_pos.copy()
#         self.uav_prev_actions.fill(0.0)
#         self.gn_energy = np.ones(NUM_GN) * GN_E_INIT
#         self.target_detect_count[:] = 0
#         self.t = 0

#         # --- 新增：重置历史记录 ---
#         self.detection_history = []

#         # 初始化 UAV 的轨迹
#         self.agent_paths = [[] for _ in range(NUM_UAV)]
#         for i in range(NUM_UAV):
#             self.agent_paths[i].append(self.uav_pos[i].copy())

#         return self._get_obs()

#     def _get_obs(self):
#         obs = []
#         for i in range(NUM_UAV):
#             x, y = self.uav_pos[i]
#             # 归一化到 [-1, 1] 范围
#             xn, yn = x / self.half_area, y / self.half_area
#             prev = self.uav_prev_actions[i]
#             # prev[0]=theta, prev[1]=dist, prev[2]=beta, prev[3]=rho (都是归一化的)
#             s = np.array([xn, yn, prev[1], prev[0], prev[2], prev[3]], dtype=np.float32)
#             obs.append(s)
#         return np.stack(obs, axis=0)

#     def step(self, actions):
#         # 动作 a \in [-1, 1]
#         actions = np.clip(np.array(actions, dtype=float), -1.0, 1.0)

#         # 实际动作 (real_actions) 存储转换后的物理值：[theta_rad, d_m, beta_[0,1], rho_[0,1]]
#         real_actions = np.zeros_like(actions)
#         for i in range(NUM_UAV):
#             theta = angle_from_norm(actions[i, 0])  # [-pi, pi]
#             d = dist_from_norm(actions[i, 1])  # [-D_MAX, D_MAX]
#             beta = beta_from_norm(actions[i, 2])  # [0, 1]
#             rho = rho_from_norm(actions[i, 3])  # [0, 1]
#             real_actions[i] = [theta, d, beta, rho]

#         # UAV 移动
#         self.uav_prev_pos = self.uav_pos.copy()
#         new_positions = self.uav_pos.copy()
#         uav_speeds = np.zeros(NUM_UAV)
#         for i in range(NUM_UAV):
#             theta, d = real_actions[i, 0], real_actions[i, 1]
#             dx = d * math.cos(theta)
#             dy = d * math.sin(theta)
#             nx = self.uav_pos[i, 0] + dx
#             ny = self.uav_pos[i, 1] + dy
#             new_positions[i] = [nx, ny]
#             # 速度 v = |d| / DT
#             uav_speeds[i] = abs(d) / DT
#         self.uav_pos = new_positions.copy()

#         # 违规检测
#         collision_flags = np.zeros(NUM_UAV, dtype=int)  # 碰撞
#         oob_flags = np.zeros(NUM_UAV, dtype=int)  # 越界

#         lower_bound = -self.half_area
#         upper_bound = self.half_area

#         for i in range(NUM_UAV):
#             x, y = self.uav_pos[i]
#             # 越界检测: 范围 [-250, 250]
#             if x < lower_bound or x > upper_bound or y < lower_bound or y > upper_bound:
#                 oob_flags[i] = 1
#                 self.uav_pos[i] = self.uav_prev_pos[i].copy()

#         for i in range(NUM_UAV):
#             for j in range(i + 1, NUM_UAV):
#                 di = np.linalg.norm(self.uav_pos[i] - self.uav_pos[j])
#                 if di < SAFE_DISTANCE:
#                     collision_flags[i] = collision_flags[j] = 1
#                     # 避免在同一时间步内多次回退
#                     if oob_flags[i] == 0: self.uav_pos[i] = self.uav_prev_pos[i].copy()
#                     if oob_flags[j] == 0: self.uav_pos[j] = self.uav_prev_pos[j].copy()

#         # GN 分配（基于定向天线约束）
#         chosen_gn = -np.ones(NUM_UAV, dtype=int)
#         gn_taken = np.zeros(NUM_GN, dtype=bool)

#         THETA_HALF_BW_RAD_GN = math.radians(THETA_HALF_BW_DEG_GN)
#         MAX_BEAM_HORIZ_DIST_GN = (H_UAV - H_GN) * math.tan(THETA_HALF_BW_RAD_GN)

#         for i in range(NUM_UAV):
#             dists = np.linalg.norm(self.gn_pos - self.uav_pos[i], axis=1)
#             order = np.argsort(dists)

#             for g in order:
#                 is_available = not gn_taken[g]
#                 is_in_beam = (dists[g] <= MAX_BEAM_HORIZ_DIST_GN)

#                 if is_in_beam and is_available:
#                     chosen_gn[i] = g
#                     gn_taken[g] = True
#                     break

#         # --- 通信干扰 ---
#         p_main = real_actions[:, 2] * P_TX  # 通信+充电功率 (beta * P_TX)
#         p_side = (1 - real_actions[:, 2]) * P_TX  # 探测功率 ((1-beta) * P_TX)
#         rho = real_actions[:, 3]  # 通信比例

#         rate = np.zeros(NUM_UAV)  # 通信速率
#         eh = np.zeros(NUM_GN)  # 地面设备收集能量速率 (速率 W/s)
#         uav_charge_gains = np.zeros(NUM_UAV) # 记录每个UAV贡献的充电量 (单位: mJ)

#         for i in range(NUM_UAV):
#             g = chosen_gn[i]
#             if g == -1:
#                 continue

#             g_main = avg_pathloss_linear(self.uav_pos[i], self.gn_pos[g])
#             interference = 0.0
#             for j in range(NUM_UAV):
#                 if j == i: continue
#                 g_int = avg_pathloss_linear(self.uav_pos[j], self.gn_pos[g])
#                 interference += p_main[j] * g_int * rho[j]

#             signal = rho[i] * p_main[i] * g_main
#             sinr = signal / (interference + ZIP0 * B + 1e-30)
#             rate[i] = B * math.log2(1 + sinr)

#             P_rx_eh = (1 - rho[i]) * p_main[i] * g_main
#             eh[g] += rf_energy_harvested(P_rx_eh) * 1e3  # 转换为 mJ/s
#             uav_charge_gains[i] = rf_energy_harvested(P_rx_eh) * 1e3 * DT

#         # GN 能量更新
#         pre_gn_energy = self.gn_energy.copy()
#         consumes = np.clip(np.random.normal(GN_BASE_CONSUME_MEAN, GN_BASE_CONSUME_STD, size=NUM_GN), 0.0, None) * DT
#         gains = eh * DT
#         self.gn_energy = np.clip(self.gn_energy - consumes + gains, 0.0, GN_E_MAX)
#         step_total_charge = np.sum(gains)

#         # 探测与干扰
#         radar_Pr = np.zeros((NUM_UAV, NUM_TARGET))
#         for i in range(NUM_UAV):
#             for n in range(NUM_TARGET):
#                 radar_Pr[i, n] = radar_received_power(self.uav_pos[i], self.target_pos[n], p_side[i])

#         valid_detection_matrix = np.zeros((NUM_UAV, NUM_TARGET), dtype=bool)
#         for i in range(NUM_UAV):
#             for n in range(NUM_TARGET):
#                 interference = 0.0
#                 for j in range(NUM_UAV):
#                     if j == i:
#                         continue
#                     inter = radar_Pr[j, n]
#                     interference += inter
#                 sinr_db = lin2db(radar_Pr[i, n] / (interference + N0 + 1e-30))

#                 if sinr_db >= MIN_SINR_DB and rate[i] >= MIN_RATE:
#                     valid_detection_matrix[i, n] = True

#         pre_target_detect_count = self.target_detect_count.copy()
#         for n in range(NUM_TARGET):
#             self.target_detect_count[n] += np.sum(valid_detection_matrix[:, n])

#         # UAV 能耗
#         uav_flight_energy_consumed = np.array([uav_flight_energy(v) for v in uav_speeds])
#         uav_comm_energy_consumed = P_TX * DT
#         uav_energy_consumed = uav_flight_energy_consumed + uav_comm_energy_consumed

#         # === 奖励计算 ===

#         # --- MODIFIED ---
#         # 1. 计算当前时隙总共发生了多少次 *探测事件*
#         #    (如果 2 个 UAV 探测了同 1 个目标，算 2 次)
#         num_targets_detected_this_slot = np.sum(valid_detection_matrix)
#         # --- MODIFIED END ---

#         sense_fair = jain_index(self.target_detect_count)
#         charge_fair = jain_index(self.gn_energy - 13.3)
#         per_agent_rewards = np.zeros(NUM_UAV)

#         for i in range(NUM_UAV):
#             for n in range(NUM_TARGET):
#                 q = pre_target_detect_count[n]
#                 if valid_detection_matrix[i, n]:
#                     r_sense_incentive = W1 * (α ** q)
#                     per_agent_rewards[i] += r_sense_incentive

#         min_gn_energy_idx = np.argmin(pre_gn_energy)
#         for i in range(NUM_UAV):
#             if chosen_gn[i] == min_gn_energy_idx:
#                 per_agent_rewards[i] += W2

#         for i in range(NUM_UAV):
#             energy_i = uav_energy_consumed[i]
            
#             # 2. 计算 r_sense 奖励
#             # (公平性 * 当前时隙总探测次数)
#             r_sense = W_SENSE * sense_fair * num_targets_detected_this_slot
            
#             r_charge = W_CHARGE * charge_fair * uav_charge_gains[i]
#             base = (r_sense + r_charge) / ((energy_i + 1e-6) * W_ENERGY)

#             penalty = 0.0
#             if collision_flags[i]:
#                 penalty += COLLISION_PENALTY
#             if oob_flags[i]:
#                 penalty += OUT_OF_BOUNDS_PENALTY

#             per_agent_rewards[i] += base - penalty
#             if chosen_gn[i] == -1:
#                 per_agent_rewards[i] += EXPLORATION_REWARD

#         for g in range(NUM_GN):
#             if self.gn_energy[g] < GN_ENERGY_THRESHOLD:
#                 dists = np.linalg.norm(self.uav_pos - self.gn_pos[g], axis=1)
#                 nearest_uav = int(np.argmin(dists))
#                 per_agent_rewards[nearest_uav] -= W_PENALTY

#         self.uav_prev_actions = actions.copy()
#         self.t += 1
#         done = (self.t >= self.episode_steps)

#         for i in range(NUM_UAV):
#             self.agent_paths[i].append(self.uav_pos[i].copy())

#         self.detection_history.append(valid_detection_matrix.copy())

#         info = {
#             'uav_pos': self.uav_pos.copy(),
#             'chosen_gn': chosen_gn.copy(),
#             'gn_energy': self.gn_energy.copy(),
#             'target_detect_count': self.target_detect_count.copy(),
#             'collision_flags': collision_flags,
#             'oob_flags': oob_flags,
#             'valid_detection_matrix': valid_detection_matrix,
#             'sense_fair': sense_fair,
#             'charge_fair': charge_fair,
#             'step_uav_energy': np.mean(uav_energy_consumed),
#             'step_total_charge': step_total_charge
#         }
#         return self._get_obs(), per_agent_rewards, done, info

#     def render(self, save_dir=None, episode_index=None, plot_links=True):
#         """
#         渲染无人机的飞行轨迹，并在每步位置标记'x'，
#         同时绘制无人机到目标的“成功探测连线”。

#         Args:
#             save_dir (str, optional): 保存图像的目录。
#             episode_index (int, optional): 回合索引，用于文件名。
#             plot_links (bool): 是否绘制探测连线。
#         """
#         plt.figure(figsize=(10, 8))
#         ax = plt.gca()

#         # 绘制地面设备（蓝点）
#         plt.scatter(self.gn_pos[:, 0], self.gn_pos[:, 1], c='blue', label='Ground Nodes (GNs)', s=50, zorder=2)

#         # 绘制目标位置（红点）
#         plt.scatter(self.target_pos[:, 0], self.target_pos[:, 1], c='red', label='Targets', s=50, zorder=2)

#         # 绘制每个无人机的起始位置（大 "X"）
#         plt.scatter(UAV_START_POS[:, 0], UAV_START_POS[:, 1], c='green', marker='X', label='UAV Start Position', s=150,
#                     zorder=3)

#         colors = ['red', 'blue', 'green']
#         for i in range(NUM_UAV):
#             # 转换为numpy数组以便于操作
#             traj = np.array(self.agent_paths[i])

#             # 1. 绘制平滑的轨迹线
#             plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.7, label=f'UAV {i + 1} Trajectory',
#                      zorder=1)

#             # 2. 在每一步的位置绘制小'x'
#             # 从第二个点开始，因为第一个点是起始点
#             if len(traj) > 1:
#                 plt.scatter(traj[1:, 0], traj[1:, 1], marker='x', color=colors[i], s=20, alpha=0.8, zorder=3)

#         # 3. 绘制成功探测连线
#         link_plotted = False
#         if plot_links and len(self.detection_history) > 0:
#             for t in range(len(self.detection_history)):
#                 valid_matrix = self.detection_history[t]

#                 for i in range(NUM_UAV):
#                     # 获取UAV在t+1时刻的位置 (因为agent_paths[0]是初始位置)
#                     uav_pos = self.agent_paths[i][t + 1]

#                     for n in range(NUM_TARGET):
#                         if valid_matrix[i, n]:
#                             target_pos = self.target_pos[n]

#                             # 绘制从UAV到目标的虚线
#                             plt.plot([uav_pos[0], target_pos[0]],
#                                      [uav_pos[1], target_pos[1]],
#                                      color=colors[i],
#                                      linestyle=':',
#                                      linewidth=0.7,
#                                      alpha=0.4,
#                                      zorder=1)
#                             link_plotted = True

#         # 设置绘图区域范围
#         plt.xlim(-self.half_area, self.half_area)
#         plt.ylim(-self.half_area, self.half_area)
#         plt.gca().set_aspect('equal', adjustable='box')  # 保持x,y轴比例一致

#         # 标签、标题、图例
#         plt.xlabel("X position (m)")
#         plt.ylabel("Y position (m)")
#         plt.title(f"UAV Trajectories and Detections (Episode {episode_index})")

#         # --- 图例处理 ---
#         handles, labels = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))

#         if plot_links and link_plotted:
#             from matplotlib.lines import Line2D
#             link_legend = Line2D([0], [0], color='gray', linestyle=':',
#                                  linewidth=1, label='Successful Detection')
#             by_label[link_legend.get_label()] = link_legend

#         plt.legend(by_label.values(), by_label.keys(),
#                    loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
#         plt.grid(True)
#         plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整布局为图例留出空间

#         # 保存图像
#         if save_dir is not None and episode_index is not None:
#             os.makedirs(save_dir, exist_ok=True)
#             file_path = os.path.join(save_dir, f"episode_{episode_index}.png")
#             plt.savefig(file_path, bbox_inches='tight', dpi=150)
#             print(f"[Render] Saved trajectory plot: {file_path}")

#         # 可选：显示图像（非阻塞）
#         plt.show(block=False)
#         plt.pause(0.1)  # 暂停一下以确保图像有时间渲染
#         plt.close()

#     def close(self):
#         plt.close('all')  # 关闭所有matplotlib窗口


# 文档版+连接失败奖励
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import gym
from gym.spaces import Box

# ------------------- 常量参数定义（含单位说明） -------------------
AREA_SIZE = 500.0  # m，环境边长
HALF_AREA = AREA_SIZE / 2.0  # m, 半边长
DT = 2.0  # s，每时间步长度

NUM_UAV = 3  # 无人机数量
NUM_TARGET = 20  # 目标数量
NUM_GN = 10  # 地面设备数量

# 无人机起始位置（m），从 [0, 500] 移到 [-250, 250]
UAV_START_POS_ORIGINAL = np.array([[100.0, 200.0], [350.0, 200.0], [200.0, 400.0]])
UAV_START_POS = UAV_START_POS_ORIGINAL - HALF_AREA

# 高度参数
H_UAV = 15.0  # m，无人机高度
H_GN = 5.0  # m，地面设备高度
H_TARGET = 0.0  # m，目标高度

# 运动限制
D_MAX = 80.0  # m，每步最大移动距离 (最大欧氏距离)
SAFE_DISTANCE = 40.0  # m，无人机最小安全距离

# 通信与感知参数
P_TX = 30.0  # W，无人机发射功率
B = 2e6  # Hz，通信带宽
FC = 3e9  # Hz，载波频率
C = 3e8  # m/s，光速
LAMBDA = C / FC  # m，波长
G_TX = 10 ** 1.3  # 无量纲，发射天线增益
G_RX = 10 ** 1.3  # 无量纲，接收天线增益
SIGMA_RCS = 0.25  # m²，雷达散射截面

KB = 1.38e-23  # J/K，玻尔兹曼常数
TEMP_K = 290.0  # K，噪声温度
NOISE_FIG_DB = 10 ** 0.5  # dB，噪声系数
N0 = KB * TEMP_K * B * NOISE_FIG_DB  # 噪声功率

# 信道参数
LOS_A = 4.88
LOS_B = 0.43
ATT_LOS = 0.1  # LoS 衰减系数
ATT_NLOS = 21  # NLoS 衰减系数
ZIP0 = 10 ** -20.4  # 噪声dBm/Hz

# 能量收集参数（文档公式）
EH_U = 24e-3
EH_r1 = 150
EH_r2 = 0.014

# GN 能量模型参数
GN_E_MAX = 25.0  # mJ，地面设备最大能量
GN_E_INIT = 20.0  # mJ，初始能量
GN_BASE_CONSUME_MEAN = 0.06  # mJ/s，平均耗能速率
GN_BASE_CONSUME_STD = 0.01  # mJ/s，耗能标准差

# UAV 飞行能耗模型参数
V_TIP = 120.0  # m/s，旋翼叶尖速度
C1 = 80.0  # W，桨叶剖面功率系数
C2 = 22.0
C3 = 263.4  # W，诱导功率系数
C4 = 0.0092  # 无量纲，寄生功率系数

# 奖励权重与惩罚
W_SENSE = 10.0
W_CHARGE = 8
W_ENERGY = 0.5
W_PENALTY = 400.0
COLLISION_PENALTY = 100.0
OUT_OF_BOUNDS_PENALTY = 100.0
W1 = 2000  # 首次探测的奖励
α = 0.05  # 衰减因子
W2 = 600  # 充电激励奖励
EXPLORATION_REWARD = 10.0  # 连接失败奖励

GN_ENERGY_THRESHOLD = 10.6  # mJ
MIN_SINR_DB = 15.0  # dB，探测 SINR 阈值
MIN_RATE = 1e5  # bps，通信速率阈值
THETA_HALF_BW_DEG_GN = 60.0  # 通信范围





# 随机种子
SEED = 318
np.random.seed(SEED)
random.seed(SEED)


# ------------------- 工具函数 -------------------
# 动作空间 a \in [-1, 1]
def angle_from_norm(theta_norm):
    """角度：[-1, 1] -> [-pi, pi]"""
    return float(theta_norm) * math.pi


def dist_from_norm(dist_norm):
    """距离：[-1, 1] -> [-D_MAX, D_MAX]"""
    return float(dist_norm) * D_MAX


def beta_from_norm(beta_norm):
    """beta: [-1, 1] -> [0, 1]"""
    return (float(beta_norm) + 1.0) / 2.0


def rho_from_norm(rho_norm):
    """rho: [-1, 1] -> [0, 1]"""
    return (float(rho_norm) + 1.0) / 2.0


def los_probability(uav_pos, gn_pos):
    """LoS 概率模型（文档公式）"""
    dx = gn_pos[0] - uav_pos[0]
    dy = gn_pos[1] - uav_pos[1]
    horiz = math.hypot(dx, dy)
    elev = math.atan2(H_UAV - H_GN, horiz)
    elev_deg = math.degrees(elev)
    p = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (elev_deg - LOS_A)))
    return p


def avg_pathloss_linear(uav_pos, gn_pos, theta_half_bw_deg=60.0):
    """平均信道功率增益（含定向天线影响）"""
    dx = gn_pos[0] - uav_pos[0]
    dy = gn_pos[1] - uav_pos[1]
    d3d = math.hypot(math.hypot(dx, dy), H_UAV - H_GN)
    p_l = los_probability(uav_pos, gn_pos)
    eta_avg = (d3d ** -2) * ((4 * math.pi * FC / C) ** -2) * ((p_l * ATT_LOS + (1 - p_l) * ATT_NLOS) ** -1)
    theta_rad = math.radians(theta_half_bw_deg)
    horiz = math.hypot(dx, dy)
    if horiz <= (H_UAV - H_GN) * math.tan(theta_rad):
        dir_gain_factor = 2.28 / (theta_rad ** 2)
        h = eta_avg * dir_gain_factor
    else:
        h = eta_avg
    return h


def radar_received_power(uav_pos, target_pos, p_side):
    """雷达回波功率（文档公式）"""
    dx = target_pos[0] - uav_pos[0]
    dy = target_pos[1] - uav_pos[1]
    R = math.hypot(math.hypot(dx, dy), H_UAV - H_TARGET)
    Pr = p_side * G_TX * G_RX * (LAMBDA ** 2 * SIGMA_RCS) / ((4.0 * math.pi) ** 3 * (R ** 4))
    return Pr


def comm_rx_power(uav_pos, gn_pos, p_main, rho):
    """通信接收功率 = ρ * p_main * g"""
    g = avg_pathloss_linear(uav_pos, gn_pos)
    P_rx = rho * p_main * g
    return float(max(P_rx, 0.0))


def rf_energy_harvested(p_rx):
    """能量收集电路模型（文档公式）"""
    val = EH_U / (1 + math.exp(EH_r1 * EH_r2)) * ((1 + math.exp(EH_r1 * EH_r2)) /
                                                  (1 + math.exp(EH_r1 * EH_r2) * math.exp(-EH_r1 * p_rx)) - 1)
    return val


def uav_flight_energy(v):
    """无人机能耗（文档物理模型）"""
    # 注意：v 是速度的绝对值
    term1 = C1 * (1.0 + 3.0 * (v ** 2) / (V_TIP ** 2))
    term2 = C2 * math.sqrt(math.sqrt(C3 + (v ** 4) / 4) - (v ** 2) / 2)
    term3 = C4 * (v ** 3)
    P_total = term1 + term2 + term3
    return P_total * DT


def lin2db(x):
    return 10.0 * math.log10(max(x, 1e-30))


def jain_index(x):
    x = np.array(x, dtype=float)
    num = (np.sum(x) ** 2)
    den = len(x) * np.sum(x ** 2) + 1e-12
    return float(num / den)


# ------------------- 环境定义 -------------------
class UAVEnvFixed:
    def __init__(self,
                 target_pos_list=None,
                 gn_pos_list=None,
                 gn_energy_init=GN_E_INIT,
                 episode_steps=80):
        self.area = AREA_SIZE
        self.half_area = HALF_AREA  # [-250, 250]
        self.episode_steps = episode_steps

        # 原始固定位置 (在 [0, 500])
        target_pos_original = np.array([
    [50.0, 120.0], [200.0, 180.0], [320.0, 380.0], [400.0, 350.0], [120.0, 450.0],
    [250.0, 400.0], [60.0, 320.0], [170.0, 350.0], [300.0, 300.0], [370.0, 450.0],
    [50.0, 100.0], [420.0, 150.0], [150.0, 240.0], [320.0, 50.0], [380.0, 120.0],
    [100.0, 400.0], [270.0, 150.0], [220.0, 220.0], [60.0, 250.0], [420.0, 70.0]
])
        gn_pos_original = np.array([
    [30.0, 220.0], [160.0, 180.0], [350.0, 110.0], [370.0, 400.0], [50.0, 90.0],
    [310.0, 330.0], [320.0, 180.0], [200.0, 280.0], [180.0, 440.0], [110.0, 380.0]
])

        # 转换为 [-250, 250] 坐标系
        self.target_pos = (target_pos_original - self.half_area) if target_pos_list is None else (
                np.array(target_pos_list) - self.half_area)
        self.gn_pos = (gn_pos_original - self.half_area) if gn_pos_list is None else (
                np.array(gn_pos_list) - self.half_area)

        # 状态初始化
        self.uav_pos = UAV_START_POS.copy()
        self.uav_prev_pos = self.uav_pos.copy()
        self.uav_prev_actions = np.zeros((NUM_UAV, 4))  # 存储的是归一化后的动作 [-1, 1]
        self.gn_energy = np.ones(NUM_GN) * gn_energy_init

        # --- 新增：预先生成整个 episode 的 GN 消耗时间表 ---
        # 1. 生成 (episode_steps, NUM_GN) 形状的随机能耗速率 (mJ/s)
        #    由于顶层设置了 SEED，这个表在每次运行时都是一样的
        base_rates_schedule = np.random.normal(
            loc=GN_BASE_CONSUME_MEAN,
            scale=GN_BASE_CONSUME_STD,
            size=(self.episode_steps, NUM_GN) 
        )
        # 2. 保存这个固定的能耗表 (裁剪确保非负)
        self.gn_consume_schedule = np.clip(base_rates_schedule, 0.0, None)
        # --- 修改结束 ---

        # # --- 修改：生成一套“固定”的随机初始能量 ---
        # # 1. 仅在 __init__ 中生成一次随机能量（例如均值为10，标准差为2）
        # initial_energies = np.random.normal(loc=gn_energy_init, scale=0.5, size=NUM_GN)
        
        # # 2. 将这套能量作为“模板”保存下来
        # self.gn_energy_initial_state = np.clip(initial_energies, 0.0, GN_E_MAX).copy()
        
        # # 3. 将当前能量设置为这个“模板”
        # self.gn_energy = self.gn_energy_initial_state.copy()
        # # --- 修改结束 ---

        self.target_detect_count = np.zeros(NUM_TARGET, dtype=int)
        self.t = 0
        self.agent_paths = [[] for _ in range(NUM_UAV)]  # 用于存储每个无人机的轨迹

        # --- 新增：为render函数存储历史探测数据 ---
        self.detection_history = []

        # --- 为 MARL 框架添加的属性 ---
        self.agent_num = NUM_UAV
        self.uav_num = NUM_UAV  # 兼容 PPO 和 MAAS

        # 观测维度 = 6 (xn, yn, prev_dist, prev_theta, prev_beta, prev_rho)
        self.observation_space_len = 6
        # 动作维度 = 4 (theta, dist, beta, rho)
        self.action_space_len = 4

        # 定义 Gym 空间 (PPO 和 DDPG/SAC 的 agents_config_dict 需要)
        # 观测空间 [-1, 1]，注意形状是 (obs_len,)
        self.observation_space = Box(-1.0, 1.0, (self.observation_space_len,))
        # 动作空间 [-1, 1]，注意形状是 (act_len,)
        self.action_space = Box(-1.0, 1.0, (self.action_space_len,))
        # --- MARL 属性添加完毕 ---

        # 轨迹初始化 (UAV_START_POS 已经在 [-250, 250] 范围内)
        for i in range(NUM_UAV):
            self.agent_paths[i].append(self.uav_pos[i].copy())

    def reset(self):
        self.uav_pos = UAV_START_POS.copy()
        self.uav_prev_pos = self.uav_pos.copy()
        self.uav_prev_actions.fill(0.0)
        self.gn_energy = np.ones(NUM_GN) * GN_E_INIT

        # # --- 修改：从“模板”恢复能量，而不是重新生成 ---
        # self.gn_energy = self.gn_energy_initial_state.copy()
        # # --- 修改结束 ---

        self.target_detect_count[:] = 0
        self.t = 0

        # --- 新增：重置历史记录 ---
        self.detection_history = []

        # 初始化 UAV 的轨迹
        self.agent_paths = [[] for _ in range(NUM_UAV)]
        for i in range(NUM_UAV):
            self.agent_paths[i].append(self.uav_pos[i].copy())

        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(NUM_UAV):
            x, y = self.uav_pos[i]
            # 归一化到 [-1, 1] 范围
            xn, yn = x / self.half_area, y / self.half_area
            prev = self.uav_prev_actions[i]
            # prev[0]=theta, prev[1]=dist, prev[2]=beta, prev[3]=rho (都是归一化的)
            s = np.array([xn, yn, prev[1], prev[0], prev[2], prev[3]], dtype=np.float32)
            obs.append(s)
        return np.stack(obs, axis=0)

    def step(self, actions):
        # 动作 a \in [-1, 1]
        actions = np.clip(np.array(actions, dtype=float), -1.0, 1.0)

        # 实际动作 (real_actions) 存储转换后的物理值：[theta_rad, d_m, beta_[0,1], rho_[0,1]]
        real_actions = np.zeros_like(actions)
        for i in range(NUM_UAV):
            theta = angle_from_norm(actions[i, 0])  # [-pi, pi]
            d = dist_from_norm(actions[i, 1])  # [-D_MAX, D_MAX]
            beta = beta_from_norm(actions[i, 2])  # [0, 1]
            rho = rho_from_norm(actions[i, 3])  # [0, 1]
            real_actions[i] = [theta, d, beta, rho]

        # UAV 移动
        self.uav_prev_pos = self.uav_pos.copy()
        new_positions = self.uav_pos.copy()
        uav_speeds = np.zeros(NUM_UAV)
        for i in range(NUM_UAV):
            theta, d = real_actions[i, 0], real_actions[i, 1]
            dx = d * math.cos(theta)
            dy = d * math.sin(theta)
            nx = self.uav_pos[i, 0] + dx
            ny = self.uav_pos[i, 1] + dy
            new_positions[i] = [nx, ny]
            # 速度 v = |d| / DT
            uav_speeds[i] = abs(d) / DT
        self.uav_pos = new_positions.copy()

        # 违规检测
        collision_flags = np.zeros(NUM_UAV, dtype=int)  # 碰撞
        oob_flags = np.zeros(NUM_UAV, dtype=int)  # 越界

        lower_bound = -self.half_area
        upper_bound = self.half_area

        for i in range(NUM_UAV):
            x, y = self.uav_pos[i]
            # 越界检测: 范围 [-250, 250]
            if x < lower_bound or x > upper_bound or y < lower_bound or y > upper_bound:
                oob_flags[i] = 1
                self.uav_pos[i] = self.uav_prev_pos[i].copy()

        for i in range(NUM_UAV):
            for j in range(i + 1, NUM_UAV):
                di = np.linalg.norm(self.uav_pos[i] - self.uav_pos[j])
                if di < SAFE_DISTANCE:
                    collision_flags[i] = collision_flags[j] = 1
                    # 避免在同一时间步内多次回退
                    if oob_flags[i] == 0: self.uav_pos[i] = self.uav_prev_pos[i].copy()
                    if oob_flags[j] == 0: self.uav_pos[j] = self.uav_prev_pos[j].copy()

        # GN 分配（基于定向天线约束）
        chosen_gn = -np.ones(NUM_UAV, dtype=int)
        gn_taken = np.zeros(NUM_GN, dtype=bool)

        THETA_HALF_BW_RAD_GN = math.radians(THETA_HALF_BW_DEG_GN)
        MAX_BEAM_HORIZ_DIST_GN = (H_UAV - H_GN) * math.tan(THETA_HALF_BW_RAD_GN)

        # if self.t == 0:
        #     print(f"[Env Info] Max horizontal distance for GN beam: {MAX_BEAM_HORIZ_DIST_GN:.2f} m")

        for i in range(NUM_UAV):
            dists = np.linalg.norm(self.gn_pos - self.uav_pos[i], axis=1)
            order = np.argsort(dists)

            for g in order:
                is_available = not gn_taken[g]
                is_in_beam = (dists[g] <= MAX_BEAM_HORIZ_DIST_GN)

                if is_in_beam and is_available:
                    chosen_gn[i] = g
                    gn_taken[g] = True
                    break

        # --- 通信干扰 ---
        p_main = real_actions[:, 2] * P_TX  # 通信+充电功率 (beta * P_TX)
        p_side = (1 - real_actions[:, 2]) * P_TX  # 探测功率 ((1-beta) * P_TX)
        rho = real_actions[:, 3]  # 通信比例

        rate = np.zeros(NUM_UAV)  # 通信速率
        eh = np.zeros(NUM_GN)  # 地面设备收集能量速率 (速率 W/s)


        for i in range(NUM_UAV):
            g = chosen_gn[i]
            if g == -1:
                continue

            g_main = avg_pathloss_linear(self.uav_pos[i], self.gn_pos[g])
            interference = 0.0
            for j in range(NUM_UAV):
                if j == i: continue
                g_int = avg_pathloss_linear(self.uav_pos[j], self.gn_pos[g])
                interference += p_main[j] * g_int * rho[j]

            signal = rho[i] * p_main[i] * g_main
            sinr = signal / (interference + ZIP0 * B + 1e-30)
            rate[i] = B * math.log2(1 + sinr)

            P_rx_eh = (1 - rho[i]) * p_main[i] * g_main
            eh[g] += rf_energy_harvested(P_rx_eh) * 1e3  # 转换为 mJ/s

        # GN 能量更新
        pre_gn_energy = self.gn_energy.copy()
        #print(self.gn_energy)
        #consumes = np.clip(np.random.normal(GN_BASE_CONSUME_MEAN, GN_BASE_CONSUME_STD, size=NUM_GN), 0.0, None) * DT
        
        # 新代码:
        # self.t 是当前时间步 (0, 1, 2, ...)
        # 我们从能耗表中查找第 t 行的数据
        current_step_rates = self.gn_consume_schedule[self.t]
        consumes = current_step_rates * DT
        # --- 修改结束 ---
        
        gains = eh * DT
        self.gn_energy = np.clip(self.gn_energy - consumes + gains, 0.0, GN_E_MAX)
        step_total_charge = np.sum(gains)

        # 探测与干扰
        radar_Pr = np.zeros((NUM_UAV, NUM_TARGET))
        for i in range(NUM_UAV):
            for n in range(NUM_TARGET):
                radar_Pr[i, n] = radar_received_power(self.uav_pos[i], self.target_pos[n], p_side[i])

        valid_detection_matrix = np.zeros((NUM_UAV, NUM_TARGET), dtype=bool)
        for i in range(NUM_UAV):
            for n in range(NUM_TARGET):
                interference = 0.0
                for j in range(NUM_UAV):
                    if j == i:
                        continue
                    inter = radar_Pr[j, n]
                    interference += inter
                sinr_db = lin2db(radar_Pr[i, n] / (interference + N0 + 1e-30))

                if sinr_db >= MIN_SINR_DB and rate[i] >= MIN_RATE:
                    valid_detection_matrix[i, n] = True

        pre_target_detect_count = self.target_detect_count.copy()
        for n in range(NUM_TARGET):
            self.target_detect_count[n] += np.sum(valid_detection_matrix[:, n])

        # UAV 能耗
        uav_flight_energy_consumed = np.array([uav_flight_energy(v) for v in uav_speeds])
        uav_comm_energy_consumed = P_TX * DT
        uav_energy_consumed = uav_flight_energy_consumed + uav_comm_energy_consumed

        # === 奖励计算 ===
        sense_fair = jain_index(self.target_detect_count)
        # charge_fair = jain_index(self.gn_energy)
        charge_fair = jain_index(self.gn_energy)
        per_agent_rewards = np.zeros(NUM_UAV)

        for i in range(NUM_UAV):
            for n in range(NUM_TARGET):
                q = pre_target_detect_count[n]
                if valid_detection_matrix[i, n]:
                    r_sense_incentive = W1 * (α ** q)
                    per_agent_rewards[i] += r_sense_incentive

        # min_gn_energy_idx = np.argmin(pre_gn_energy)
        # for i in range(NUM_UAV):
        #     if chosen_gn[i] == min_gn_energy_idx:
        #         per_agent_rewards[i] += W2

        # --- 修改：激励奖励给电量倒数三名的GN ---
        # 1. 获取充电前 (pre_gn_energy) 电量倒数三名的 GN 索引
        #    我们使用 argsort 来获取排序后的索引，然后取前3个
        
        # 定义要激励的GN数量
        K_LOWEST_GN = 3 
        
        # 确保 K_LOWEST_GN 不超过总GN数，以防配置错误
        num_to_incentivize = min(K_LOWEST_GN, NUM_GN) 
        
        # pre_gn_energy.argsort() 返回的是从小到大排序的索引
        bottom_k_gn_indices = np.argsort(pre_gn_energy)[:num_to_incentivize]

        # 2. 遍历所有 UAV，检查其是否在为这 K 个GN充电
        for i in range(NUM_UAV):
            # chosen_gn[i] 是 UAV i 选择的GN索引
            # 检查这个索引是否在“倒数K名”的列表中
            if chosen_gn[i] in bottom_k_gn_indices:
                g = chosen_gn[i]
                # eh[g] 是一个标量数字
                energy_rate_delivered = eh[g]
                # 乘以时间 DT 得到本时隙的总能量 (mJ)
                energy_delivered_mj = energy_rate_delivered * DT

                # 奖励值 = 基础权重 * 实际充电量
                incentive_reward = W2 * energy_delivered_mj
                # 如果是，则给予激励奖励 W2
                per_agent_rewards[i] += incentive_reward

                # 如果是，则给予激励奖励 W2
                # per_agent_rewards[i] += W2
        # --- 修改结束 ---


        # 原版
        for i in range(NUM_UAV):
            energy_i = uav_energy_consumed[i]
            r_sense = W_SENSE * sense_fair * np.sum(self.target_detect_count)
            r_charge = W_CHARGE * charge_fair * np.sum(self.gn_energy)
            base = (r_sense + r_charge) / ((energy_i + 1e-6) * W_ENERGY)

            penalty = 0.0
            if collision_flags[i]:
                penalty += COLLISION_PENALTY
            if oob_flags[i]:
                penalty += OUT_OF_BOUNDS_PENALTY

            if chosen_gn[i] == -1:
                per_agent_rewards[i] -= EXPLORATION_REWARD            

            per_agent_rewards[i] += base - penalty
        
        # # 激励奖励加到分子上
        # for i in range(NUM_UAV):
        #     energy_i = uav_energy_consumed[i]
        #     r_sense = W_SENSE * sense_fair * np.sum(self.target_detect_count)
        #     r_charge = W_CHARGE * charge_fair * np.sum(self.gn_energy)
        #     #base = (r_sense + r_charge) / ((energy_i + 1e-6) * W_ENERGY)

        #     penalty = 0.0
        #     if collision_flags[i]:
        #         penalty += COLLISION_PENALTY
        #     if oob_flags[i]:
        #         penalty += OUT_OF_BOUNDS_PENALTY

        #     if chosen_gn[i] == -1:
        #         per_agent_rewards[i] -= EXPLORATION_REWARD            

        #     per_agent_rewards[i] += r_sense + r_charge 
        #     per_agent_rewards[i] = (per_agent_rewards[i] - penalty) / ((energy_i + 1e-6) * W_ENERGY)

        # # 减去能耗
        # for i in range(NUM_UAV):
        #     energy_i = uav_energy_consumed[i]
        #     r_sense = W_SENSE * sense_fair * np.sum(self.target_detect_count)
        #     r_charge = W_CHARGE * charge_fair * np.sum(self.gn_energy)
        #     #base = (r_sense + r_charge) / ((energy_i + 1e-6) * W_ENERGY)
        #     base = r_sense + r_charge

        #     penalty = 0.0
        #     if collision_flags[i]:
        #         penalty += COLLISION_PENALTY
        #     if oob_flags[i]:
        #         penalty += OUT_OF_BOUNDS_PENALTY

        #     # if chosen_gn[i] == -1:
        #     #     per_agent_rewards[i] += EXPLORATION_REWARD            

        #     per_agent_rewards[i] += base - penalty -  ((energy_i + 1e-6) * W_ENERGY)
        
        

        for g in range(NUM_GN):
            if self.gn_energy[g] < GN_ENERGY_THRESHOLD:
                dists = np.linalg.norm(self.uav_pos - self.gn_pos[g], axis=1)
                nearest_uav = int(np.argmin(dists))
                per_agent_rewards[nearest_uav] -= W_PENALTY

        #total_reward = np.mean(per_agent_rewards)

        self.uav_prev_actions = actions.copy()
        self.t += 1
        done = (self.t >= self.episode_steps)

        for i in range(NUM_UAV):
            self.agent_paths[i].append(self.uav_pos[i].copy())

        # --- 新增：存储当前步的探测结果 ---
        self.detection_history.append(valid_detection_matrix.copy())

        info = {
            'uav_pos': self.uav_pos.copy(),
            'chosen_gn': chosen_gn.copy(),
            'gn_energy': self.gn_energy.copy(),
            'target_detect_count': self.target_detect_count.copy(),
            'collision_flags': collision_flags,
            'oob_flags': oob_flags,
            'valid_detection_matrix': valid_detection_matrix,
            'sense_fair': sense_fair,
            'charge_fair': charge_fair,
            'step_uav_energy': np.mean(uav_energy_consumed),
            'step_total_charge': step_total_charge
        }
        return self._get_obs(), per_agent_rewards, done, info

    def render(self, save_dir=None, episode_index=None, plot_links=True):
        """
        渲染无人机的飞行轨迹，并在每步位置标记'x'，
        同时绘制无人机到目标的“成功探测连线”。

        Args:
            save_dir (str, optional): 保存图像的目录。
            episode_index (int, optional): 回合索引，用于文件名。
            plot_links (bool): 是否绘制探测连线。
        """
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # 绘制地面设备（蓝点）
        plt.scatter(self.gn_pos[:, 0], self.gn_pos[:, 1], c='blue', label='Ground Nodes (GNs)', s=50, zorder=2)

        # 绘制目标位置（红点）
        plt.scatter(self.target_pos[:, 0], self.target_pos[:, 1], c='red', label='Targets', s=50, zorder=2)

        # 绘制每个无人机的起始位置（大 "X"）
        plt.scatter(UAV_START_POS[:, 0], UAV_START_POS[:, 1], c='green', marker='X', label='UAV Start Position', s=150,
                    zorder=3)

        colors = ['red', 'blue', 'green']
        for i in range(NUM_UAV):
            # 转换为numpy数组以便于操作
            traj = np.array(self.agent_paths[i])

            # 1. 绘制平滑的轨迹线
            plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.7, label=f'UAV {i + 1} Trajectory',
                     zorder=1)

            # 2. 在每一步的位置绘制小'x'
            # 从第二个点开始，因为第一个点是起始点
            if len(traj) > 1:
                plt.scatter(traj[1:, 0], traj[1:, 1], marker='x', color=colors[i], s=20, alpha=0.8, zorder=3)

        # 3. 绘制成功探测连线
        link_plotted = False
        if plot_links and len(self.detection_history) > 0:
            for t in range(len(self.detection_history)):
                valid_matrix = self.detection_history[t]

                for i in range(NUM_UAV):
                    # 获取UAV在t+1时刻的位置 (因为agent_paths[0]是初始位置)
                    uav_pos = self.agent_paths[i][t + 1]

                    for n in range(NUM_TARGET):
                        if valid_matrix[i, n]:
                            target_pos = self.target_pos[n]

                            # 绘制从UAV到目标的虚线
                            plt.plot([uav_pos[0], target_pos[0]],
                                     [uav_pos[1], target_pos[1]],
                                     color=colors[i],
                                     linestyle=':',
                                     linewidth=0.7,
                                     alpha=0.4,
                                     zorder=1)
                            link_plotted = True

        # 设置绘图区域范围
        plt.xlim(-self.half_area, self.half_area)
        plt.ylim(-self.half_area, self.half_area)
        plt.gca().set_aspect('equal', adjustable='box')  # 保持x,y轴比例一致

        # 标签、标题、图例
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.title(f"UAV Trajectories and Detections (Episode {episode_index})")

        # --- 图例处理 ---
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        if plot_links and link_plotted:
            from matplotlib.lines import Line2D
            link_legend = Line2D([0], [0], color='gray', linestyle=':',
                                 linewidth=1, label='Successful Detection')
            by_label[link_legend.get_label()] = link_legend

        plt.legend(by_label.values(), by_label.keys(),
                   loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整布局为图例留出空间

        # 保存图像
        if save_dir is not None and episode_index is not None:
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f"episode_{episode_index}.png")
            plt.savefig(file_path, bbox_inches='tight', dpi=150)
            print(f"[Render] Saved trajectory plot: {file_path}")

        # 可选：显示图像（非阻塞）
        plt.show(block=False)
        plt.pause(0.1)  # 暂停一下以确保图像有时间渲染
        plt.close()

    def close(self):
        plt.close('all')  # 关闭所有matplotlib窗口