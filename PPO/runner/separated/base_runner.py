# import time
# import os
# import numpy as np
# from itertools import chain
# import torch
# from tensorboardX import SummaryWriter
#
# from PPO.utils.separated_buffer import SeparatedReplayBuffer
# from PPO.utils.util import update_linear_schedule
#
#
# def _t2n(x):
#     return x.detach().cpu().numpy()
#
#
# class Runner(object):
#     def __init__(self, config):
#         self.all_args = config["all_args"]
#         self.envs = config["envs"]
#         self.eval_envs = config["eval_envs"]
#         self.device = config["device"]
#         self.num_agents = config["num_agents"]
#
#         # parameters
#         self.env_name = self.all_args.env_name
#         self.algorithm_name = self.all_args.algorithm_name
#         self.experiment_name = self.all_args.experiment_name
#         self.use_centralized_V = self.all_args.use_centralized_V
#         self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
#         self.num_env_steps = self.all_args.num_env_steps
#         self.episode_length = self.all_args.episode_length
#         self.n_rollout_threads = self.all_args.n_rollout_threads
#         self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
#         self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
#         self.hidden_size = self.all_args.hidden_size
#         self.use_render = self.all_args.use_render
#         self.recurrent_N = self.all_args.recurrent_N
#
#         # interval
#         self.save_interval = self.all_args.save_interval
#         self.use_eval = self.all_args.use_eval
#         self.eval_interval = self.all_args.eval_interval
#         self.log_interval = self.all_args.log_interval
#
#         # dir
#         self.model_dir = self.all_args.model_dir
#
#         if self.use_render:
#             import imageio
#
#             self.run_dir = config["run_dir"]
#             self.gif_dir = str(self.run_dir / "gifs")
#             if not os.path.exists(self.gif_dir):
#                 os.makedirs(self.gif_dir)
#         else:
#             # if self.use_wandb:
#             #     self.save_dir = str(wandb.run.dir)
#             # else:
#             self.run_dir = config["run_dir"]
#             self.log_dir = str(self.run_dir / "logs")
#             if not os.path.exists(self.log_dir):
#                 os.makedirs(self.log_dir)
#             self.writter = SummaryWriter(self.log_dir)
#             self.save_dir = str(self.run_dir / "models")
#             if not os.path.exists(self.save_dir):
#                 os.makedirs(self.save_dir)
#
#         from PPO.algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
#         from PPO.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy
#
#         self.policy = []
#         for agent_id in range(self.num_agents):
#             share_observation_space = (
#                 self.envs.share_observation_space[agent_id]
#                 if self.use_centralized_V
#                 else self.envs.observation_space[agent_id]
#             )
#             # policy network
#             po = Policy(
#                 self.all_args,
#                 self.envs.observation_space[agent_id],
#                 share_observation_space,
#                 self.envs.action_space[agent_id],
#                 device=self.device,
#             )
#             self.policy.append(po)
#
#         if self.model_dir is not None:
#             self.restore()
#
#         self.trainer = []
#         self.buffer = []
#         for agent_id in range(self.num_agents):
#             # algorithm
#             tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
#             # buffer
#             share_observation_space = (
#                 self.envs.share_observation_space[agent_id]
#                 if self.use_centralized_V
#                 else self.envs.observation_space[agent_id]
#             )
#             bu = SeparatedReplayBuffer(
#                 self.all_args,
#                 self.envs.observation_space[agent_id],
#                 share_observation_space,
#                 self.envs.action_space[agent_id],
#             )
#             self.buffer.append(bu)
#             self.trainer.append(tr)
#
#     def run(self):
#         raise NotImplementedError
#
#     def warmup(self):
#         raise NotImplementedError
#
#     def collect(self, step):
#         raise NotImplementedError
#
#     def insert(self, data):
#         raise NotImplementedError
#
#     @torch.no_grad()
#     def compute(self):
#         for agent_id in range(self.num_agents):
#             self.trainer[agent_id].prep_rollout()
#             next_value = self.trainer[agent_id].policy.get_values(
#                 self.buffer[agent_id].share_obs[-1],
#                 self.buffer[agent_id].rnn_states_critic[-1],
#                 self.buffer[agent_id].masks[-1],
#             )
#             next_value = _t2n(next_value)
#             self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)
#
#     def train(self):
#         train_infos = []
#         for agent_id in range(self.num_agents):
#             self.trainer[agent_id].prep_training()
#             train_info = self.trainer[agent_id].train(self.buffer[agent_id])
#             train_infos.append(train_info)
#             self.buffer[agent_id].after_update()
#
#         return train_infos
#
#     def save(self):
#         for agent_id in range(self.num_agents):
#             policy_actor = self.trainer[agent_id].policy.actor
#             torch.save(
#                 policy_actor.state_dict(),
#                 str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
#             )
#             policy_critic = self.trainer[agent_id].policy.critic
#             torch.save(
#                 policy_critic.state_dict(),
#                 str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt",
#             )
#
#     def restore(self):
#         for agent_id in range(self.num_agents):
#             policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor_agent" + str(agent_id) + ".pt")
#             self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
#             policy_critic_state_dict = torch.load(
#                 str(self.model_dir) + "/critic_agent" + str(agent_id) + ".pt"
#             )
#             self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
#
#     def log_train(self, train_infos, total_num_steps):
#         pass
#         # for agent_id in range(self.num_agents):
#         #     for k, v in train_infos[agent_id].items():
#         #         agent_k = "agent%i/" % agent_id + k
#         #         # if self.use_wandb:
#         #         #     pass
#         #         # wandb.log({agent_k: v}, step=total_num_steps)
#         #         # else:
#         #         self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
#
#     def log_env(self, env_infos, total_num_steps):
#         for k, v in env_infos.items():
#             if len(v) > 0:
#                 # if self.use_wandb:
#                 #     wandb.log({k: np.mean(v)}, step=total_num_steps)
#                 # else:
#                 self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
# 在 PPO/runner/separated/base_runner.py

import time
import numpy as np
import torch
from PPO.utils.separated_buffer import SeparatedReplayBuffer # 确认 Buffer 类被导入
# from PPO.runner.separated.base_runner import Runner as BaseRunner # 如果是修改现有文件，避免循环导入
import os # 用于路径操作

class Runner: # 可以直接修改这个类，或者继承它然后重写 run 方法
    def __init__(self, config):
        # --- 大部分初始化代码保持不变，但要适配单环境 ---
        self.all_args = config['all_args']
        self.envs = config['envs'] # <--- 现在是单个环境实例
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.run_dir = config['run_dir']
        self.writer = config['writer'] # <--- 获取 writer

        # --- 确保观察和动作空间被正确设置 ---
        # 最好在 train.py 中完成，这里再次确认
        self.envs.observation_space # 访问一下以确保初始化
        self.envs.action_space
        # 假设所有智能体都是同构的
        self.observation_space = self.envs.observation_space
        self.action_space = self.envs.action_space

        # --- 策略和 Buffer 初始化 ---
        # (这部分代码基本不变，但要确保传入正确的 space)
        #print(os.path.exists('D:/ISCPTcode/uav_irs-master/uav_irs-master/PPO/algorithms/algorithm/r_mappo.py'))

        from PPO.algorithms.algorithm.r_mappo import R_MAPPO as TrainAlgo
        from PPO.algorithms.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.observation_space # PPO 中 Critic 通常只用局部观察
            if self.all_args.use_centralized_V:
                 # 如果使用中心化 Critic，可能需要从 env 获取全局状态，这里简化处理
                 print("Warning: use_centralized_V is True, but using local observation space for critic. Adapt if global state is needed.")
                 share_observation_space = self.observation_space

            po = Policy(self.all_args,
                        self.observation_space,
                        share_observation_space,
                        self.action_space,
                        device=self.device)
            self.policy.append(po)

        if self.all_args.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            share_observation_space = self.observation_space # 同上
            if self.all_args.use_centralized_V:
                share_observation_space = self.observation_space
            # Buffer 初始化
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.observation_space,
                                       share_observation_space,
                                       self.action_space)
            self.trainer.append(tr)
            self.buffer.append(bu)

        # --- 添加回合和步数跟踪 ---
        self.episode = 0
        self.total_env_steps = 0


    def run(self):
        """ 主训练循环，按回合进行 """
        print(f"开始训练，总步数目标: {self.all_args.num_env_steps}")
        start = time.time()

        while self.total_env_steps < self.all_args.num_env_steps:
            # --- 回合开始 ---
            obs_n = self.envs.reset() # (num_agents, obs_dim) numpy 数组
            # 如果使用 RNN，初始化隐藏状态
            rnn_states = np.zeros((self.num_agents, self.all_args.recurrent_N, self.all_args.hidden_size), dtype=np.float32)
            masks = np.ones((self.num_agents, 1), dtype=np.float32) # 表示回合开始

            # --- 回合数据记录 ---
            episode_rewards_list = [] # 存储每一步每个 agent 的奖励
            total_episode_uav_energy = 0.0
            total_episode_charge = 0.0
            last_info = {}

            # --- 在一个回合内循环 ---
            for step in range(self.all_args.episode_length):
                # --- 与环境交互 ---
                # 将观察转换为 Tensor
                torch_obs_n = torch.from_numpy(obs_n).float().to(self.device)
                torch_rnn_states = torch.from_numpy(rnn_states).float().to(self.device)
                torch_masks = torch.from_numpy(masks).float().to(self.device)

                # 从策略网络获取动作
                with torch.no_grad():
                    values_n, actions_n, action_log_probs_n, rnn_states_n_out = self.collect(torch_obs_n, torch_rnn_states, torch_masks)

                # 将动作转为 numpy 以便输入环境
                cpu_actions_n = actions_n.cpu().numpy()
                # 将 RNN 状态转回 numpy
                rnn_states = rnn_states_n_out.cpu().numpy()

                # 执行动作，获取环境反馈
                next_obs_n, reward_n, done, info = self.envs.step(cpu_actions_n)

                # --- 记录回合数据 ---
                episode_rewards_list.append(reward_n) # reward_n 是 (num_agents,)
                total_episode_uav_energy += info.get('step_uav_energy', 0.0)
                total_episode_charge += info.get('step_total_charge', 0.0)
                last_info = info # 保存最后一步的 info

                # --- 存储经验到 Buffer ---
                # dones 需要是 (num_agents, 1) 的形状
                dones_n = np.full((self.num_agents, 1), done, dtype=np.float32)
                # 准备数据 (确保形状符合 Buffer 要求)
                data = (obs_n, # 当前观察 (numpy)
                        rnn_states, # 当前 RNN 状态 (numpy)
                        cpu_actions_n, # 采取的动作 (numpy)
                        action_log_probs_n.cpu().numpy(), # 动作 log 概率 (numpy)
                        values_n.cpu().numpy(), # 价值估计 (numpy)
                        reward_n.reshape(self.num_agents, 1), # 奖励 (numpy, reshape)
                        dones_n, # 完成标志 (numpy)
                        masks # 当前 mask (numpy)
                       )
                self.insert(data) # 插入 Buffer

                # --- 更新状态 ---
                obs_n = next_obs_n
                masks = np.ones((self.num_agents, 1), dtype=np.float32) * (1 - dones_n) # 如果 done=True, mask 变为 0
                self.total_env_steps += 1 # 更新总步数

                if done:
                    break # 回合结束，跳出 step 循环

            # --- 回合结束 ---
            self.episode += 1

            # --- PPO 训练步骤 ---
            self.compute_returns() # 计算回报和优势
            train_infos = self.train_policy() # 执行训练更新 (忽略返回值中的 loss)

            # --- TensorBoard 日志记录 (与 SAC 类似) ---
            # 计算回合总奖励 (按 agent 求和，再跨 agent 求平均)
            total_episode_reward = np.mean(np.sum(episode_rewards_list, axis=0))

            self.writer.add_scalar('Reward/Train', total_episode_reward, self.episode)

            if last_info: # 记录最后一步的指标
                final_sense_fair = last_info.get('sense_fair', 0.0)
                final_charge_fair = last_info.get('charge_fair', 0.0)
                final_gn_energy_list = last_info.get('gn_energy', [0.0])
                final_avg_gn_energy = np.mean(final_gn_energy_list)

                self.writer.add_scalar('Metrics/Sensing_Fairness', final_sense_fair, self.episode)
                self.writer.add_scalar('Metrics/Charging_Fairness', final_charge_fair, self.episode)
                self.writer.add_scalar('Metrics/Avg_GN_Energy', final_avg_gn_energy, self.episode)

            # 记录累积指标
            self.writer.add_scalar('Metrics/Total_UAV_Energy', total_episode_uav_energy, self.episode)
            self.writer.add_scalar('Metrics/Total_GN_Charge_mJ', total_episode_charge, self.episode)
            # --- 日志记录结束 ---

            # --- 轨迹渲染 (与 SAC 类似) ---
            if self.all_args.use_render and self.episode % self.all_args.render_interval == 0 and self.episode > 0:
                print(f"[渲染] 回合 {self.episode} - 奖励: {total_episode_reward:.3f}")
                try:
                    log_dir_path = self.writer.log_dir
                    run_id = os.path.basename(log_dir_path.strip('/'))
                    if not run_id: run_id = "ppo_default"
                    render_save_dir = f"./trajectory_plots/{run_id}" # 注意这里的相对路径是相对于 PPO 目录
                except Exception as e:
                    print(f"警告: 解析 log_dir 创建渲染路径失败. 使用默认路径. 错误: {e}")
                    render_save_dir = "./trajectory_plots/ppo_default"

                print(f"[渲染] 保存轨迹到 {render_save_dir}")
                self.envs.render(save_dir=render_save_dir, episode_index=self.episode)
            # --- 渲染结束 ---

            # --- 定期打印训练信息 ---
            if self.episode % self.all_args.log_interval == 0:
                 end = time.time()
                 print(f"回合 {self.episode}/{int(self.all_args.num_env_steps/self.all_args.episode_length)} | "
                       f"总步数 {self.total_env_steps}/{self.all_args.num_env_steps} | "
                       f"FPS {int(self.total_env_steps / (end - start))} | "
                       f"最近回合奖励 {total_episode_reward:.3f}")

            # --- 定期保存模型 ---
            if self.episode % self.all_args.save_interval == 0 or self.episode >= int(self.all_args.num_env_steps / self.all_args.episode_length):
                 self.save()

    # --- 需要修改/确认 Helper 函数 ---

    def collect(self, torch_obs_n, torch_rnn_states, torch_masks):
        """ 从策略网络收集动作、价值等信息 """
        # 这个函数需要遍历每个 agent 的 policy
        values_collector = []
        actions_collector = []
        action_log_probs_collector = []
        rnn_states_collector = [] # 存储每个 agent 输出的 RNN 状态

        for agent_id in range(self.num_agents):
            # 注意：这里的 share_obs 可能需要根据 Critic 类型调整，现在假设只用 obs
            # PPO Policy 通常需要 obs, rnn_state, mask 输入
            value, action, action_log_prob, rnn_state_out = self.policy[agent_id].act(
                torch_obs_n[agent_id].unsqueeze(0), # 需要添加 batch 维度
                torch_rnn_states[agent_id].unsqueeze(0),
                torch_masks[agent_id].unsqueeze(0)
            )
            values_collector.append(value)
            actions_collector.append(action)
            action_log_probs_collector.append(action_log_prob)
            rnn_states_collector.append(rnn_state_out)

        # 合并结果 (去掉添加的 batch 维度)
        values_n = torch.cat(values_collector, dim=0)
        actions_n = torch.cat(actions_collector, dim=0)
        action_log_probs_n = torch.cat(action_log_probs_collector, dim=0)
        rnn_states_n_out = torch.cat(rnn_states_collector, dim=0)

        return values_n, actions_n, action_log_probs_n, rnn_states_n_out


    def insert(self, data):
        """ 将收集到的数据插入到每个 agent 的 Buffer 中 """
        obs, rnn_states, actions, action_log_probs, values, rewards, dones, masks = data

        # 分别插入每个 agent 的 buffer
        # 注意：share_obs 在 PPO 里通常与 obs 相同，或中心化 Critic 时需要全局信息
        # 这里简化处理，假设 share_obs 就是 obs
        share_obs = obs # 简化假设

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[agent_id], # PPO buffer 需要 share_obs
                                         obs[agent_id],
                                         rnn_states[agent_id],
                                         actions[agent_id],
                                         action_log_probs[agent_id],
                                         values[agent_id],
                                         rewards[agent_id],
                                         masks[agent_id]) # PPO buffer 需要 mask 而不是 done

    def compute_returns(self):
         """ 计算回报和优势 """
         for agent_id in range(self.num_agents):
             # Buffer 需要知道下一个 value 来计算 GAE
             # 在回合结束时，下一个 value 通常是 0
             # 注意：这里的 share_obs 同样简化为 obs
             next_value = 0.0 # 假设回合结束
             # 需要获取当前 agent 最后的观察值
             last_obs = torch.from_numpy(self.buffer[agent_id].obs[-1]).float().unsqueeze(0).to(self.device)
             last_rnn_state = torch.from_numpy(self.buffer[agent_id].rnn_states[-1]).float().unsqueeze(0).to(self.device)
             last_mask = torch.from_numpy(self.buffer[agent_id].masks[-1]).float().unsqueeze(0).to(self.device)

             # 如果没有结束，可以从 Critic 获取 next_value
             # if not np.all(self.buffer[agent_id].masks[-1] == 0): # 检查最后一步是否为回合结束
             #     with torch.no_grad():
             #         next_value = self.policy[agent_id].get_values(last_obs, last_rnn_state, last_mask)
             #         next_value = next_value.cpu().numpy().item() # 获取标量值

             self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)


    def train_policy(self):
        """ 触发每个 agent 的训练 """
        train_infos = [] # 这个实现里似乎不需要记录 loss
        for agent_id in range(self.num_agents):
            # 从 Buffer 获取训练数据
            # PPO Trainer 需要调用 train 方法，传入 buffer
            # r_mappo.py 的 train 方法会处理数据获取和更新
             train_info = self.trainer[agent_id].train(self.buffer[agent_id])
             # train_infos.append(train_info) # 暂时不记录详细 loss
             self.buffer[agent_id].after_update() # 清空或重置 Buffer
        return train_infos # 返回空的列表或简化信息


    # save, restore, log_train, log_env 方法基本可以保持不变或稍作调整
    # 需要确保 save/restore 保存/加载 Actor 和 Critic 网络
    def save(self):
        """保存模型参数。"""
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        """加载模型参数。"""
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.all_args.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            # if not self.all_args.use_render: # 只有在非渲染模式下才加载 Critic？按需修改
            policy_critic_state_dict = torch.load(str(self.all_args.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    # 其他方法 (log_train, log_env) 可以简化或移除，因为日志记录在 run 方法中处理了