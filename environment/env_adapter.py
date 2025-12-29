import gym
import numpy as np
from gym.spaces import Box

# 1. 从您保存的文件中导入您的环境和常量
try:
    from env import UAVEnvFixed, NUM_UAV, GN_E_MAX
except ImportError:
    print("错误：请确保您的环境代码保存在 'env.py' 文件中。")
    exit()

# 2. 从您的代码中定义维度
# 观测维度：[xn, yn, prev[1], prev[0], prev[2], prev[3]]
OBS_DIM = 6
# 动作维度：[theta, d, beta, rho]
ACT_DIM = 4


class UAVEnvAdapter:
    """
    这个包装器类将您的 UAVEnvFixed 环境的 API
    适配为 light_mappo 期望的 API。
    """

    def __init__(self):
        # 1. 实例化您自己的环境
        self.env = UAVEnvFixed()

        # 2. 定义算法库(Runner)需要的属性
        self.agent_num = NUM_UAV
        self.n_agents = NUM_UAV  # 兼容不同算法的命名

        # 3. 定义观测和动作维度
        self.obs_dim = OBS_DIM
        self.action_dim = ACT_DIM

        # 4. 定义全局状态维度 (MARL Critic需要)
        # 简单起见，我们将所有智能体的个体观测拼接起来作为全局状态
        self.state_dim = self.obs_dim * self.agent_num  # 6 * 3 = 18

        # 5. 定义 Gym 空间 (算法库的核心要求)

        # 个体观测空间 (Actor使用)
        # 您的 _get_obs() 返回值都在 [-1, 1] 范围内
        obs_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        # 个体动作空间 (Actor使用)
        # 您的动作输入在 [-1, 1] 范围内
        act_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        # 全局状态空间 (Critic使用)
        state_space = Box(low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)

        # --- 提供 MAPPO 期望的 API ---
        # 算法库期望这些是列表(list)，每个智能体一个
        self.observation_space = [obs_space for _ in range(self.agent_num)]
        self.action_space = [act_space for _ in range(self.agent_num)]

        # 'share_observation_space' 是 MAPPO 对全局状态的叫法
        self.share_observation_space = [state_space for _ in range(self.agent_num)]

    def get_obs(self):
        """
        获取一个体观测的列表 (list of obs)
        """
        obs_array = self.env._get_obs()  # 您的函数返回 (3, 6) 的数组
        # 转换为列表
        return [obs_array[i] for i in range(self.agent_num)]

    def get_state(self):
        """
        获取全局状态 (一个扁平化的数组)
        """
        # _get_obs() 返回 (3, 6) 数组, flatten() 变为 (18,)
        return self.env._get_obs().flatten()

    def reset(self):
        """
        适配 reset() 函数
        您的 env.reset() 返回 (3, 6) 的数组
        MAPPO 期望返回 [obs1, obs2, obs3] 的列表
        """
        obs_array = self.env.reset()
        # 转换为列表
        return [obs_array[i] for i in range(self.agent_num)]

    def step(self, actions_list):
        """
        适配 step() 函数

        输入: actions_list (来自MAPPO, 是一个列表 [act1, act2, act3])

        返回: [list_obs, list_rewards, list_dones, list_info] (MAPPO期望的格式)
        """

        # 1. [数据转换] 将动作列表
        #    转换为您的环境期望的 (3, 4) NumPy 数组
        actions_array = np.stack(actions_list, axis=0)

        # 2. [调用您的环境]
        obs_array, per_agent_rewards, total_reward, done, info = self.env.step(actions_array)

        # 3. [数据转换] 将您的环境的返回结果
        #    转换为 MAPPO 期望的列表格式

        # 转换观测
        list_obs = [obs_array[i] for i in range(self.agent_num)]

        # 转换奖励 (MAPPO 期望一个 "列表的列表")
        list_rewards = [[r] for r in per_agent_rewards]

        # 转换 Done (您的 'done' 是全局的，复制给每个智能体)
        list_dones = [done for _ in range(self.agent_num)]

        # 转换 Info
        list_info = [{} for _ in range(self.agent_num)]  # 简单处理

        return list_obs, list_rewards, list_dones, list_info

    # --- 其他辅助函数 ---
    def render(self, mode="human"):
        # 算法库可能会调用 render，放一个占位符
        print("Render call (pass).")
        pass

    def close(self):
        self.env.close()