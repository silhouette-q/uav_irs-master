# import os
# import time
#
# import numpy as np
# import torch
# from tqdm import tqdm
#
# from Base_Agent import Base_Agent
# from SAC.SACAgent import SACAgent
# from utilities.data_structures.Replay_Buffer import Replay_Buffer
# from utilities.plot_uav_positions import plot_uav_positions
#
# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
# TRAINING_EPISODES_PER_EVAL_EPISODE = 10
# EPSILON = 1e-6
#
#
# class MASAC(Base_Agent):
#     """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
#       https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
#       to maximise the entropy of their actions as well as their cumulative reward"""
#     agent_name = "MASAC"
#
#     def __init__(self, config):
#         Base_Agent.__init__(self, config)
#         assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
#         assert self.config.hyperparameters["Actor"][
#                    "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
#         self.hyperparameters = config.hyperparameters
#         self.environment = config.environment
#         self.agents = [
#             SACAgent(config, i, d['obs_space'].shape[0], d['action_space'].shape[0], agent_num=config.agent_num) for
#             i, d in
#             enumerate(config.agents_config_dict)]
#         self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
#         self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
#                                     self.config.seed)
#
#         # 在 self.episode_number = 0 下方添加
#         self.total_episode_uav_energy = 0.0
#         self.total_episode_charge = 0.0
#
#         for a in self.agents:
#             a.parent = self
#
#     def reset_game(self):
#         """Resets the game information so we are ready to play a new episode"""
#         Base_Agent.reset_game(self)
#
#     def step(self):
#         """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
#         reward_sum = 0
#         while not self.done:
#             if self.global_step_number < self.config.hyperparameters['min_steps_before_learning']:
#                 self.actions = np.random.random((len(self.agents), 4)) * 2 - 1
#             else:
#                 self.actions = self.pick_action(False)
#             self.conduct_action(self.actions)
#             reward_sum += self.reward
#             if self.config.add_bonus:
#                 self.reward = self.reward + self.get_bonus(self.actions)
#             if self.time_for_critic_and_actor_to_learn():
#                 for i in range(self.hyperparameters["learning_updates_per_learning_session"]):
#                     self.config.learning_updates = i
#                     self.learn()
#             mask = self.done
#             self.save_experience(
#                 experience=(self.state, self.actions, self.reward, self.next_state, mask))
#             self.state = self.next_state
#             self.global_step_number += 1
#         self.episode_number += 1
#         self.config.writer.add_scalar('immediate_reward', reward_sum.sum(), self.episode_number)
#
#     def produce_actions(self, states):
#         # 验证下是否需要采样
#         all_actor_acs = []
#         for i, a in enumerate(self.agents):
#             all_actor_acs.append(a.produce_action_and_action_info(states[:, i, :]))
#         return all_actor_acs
#
#     def save_experience(self, memory=None, experience=None):
#         """Saves the recent experience to the memory buffer"""
#         if memory is None: memory = self.memory
#         if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
#         memory.add_experience(*experience)
#
#     def sample_experiences(self):
#         return self.memory.sample()
#
#     def learn(self):
#         state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
#         for i in range(len(self.agents)):
#             self.agents[i].learn(experiences=(state_batch, action_batch, reward_batch, next_state_batch, mask_batch))
#
#     def conduct_action(self, actions, eval=False):
#         self.next_state, self.reward, self.done, _ = self.environment.step(actions.cpu().numpy())
#         self.total_episode_score_so_far += sum(self.reward)
#         if self.hyperparameters["clip_rewards"]: self.reward = max(min(self.reward, 1.0), -1.0)
#
#     def get_bonus(self, actions):
#         bonus = torch.zeros_like(self.reward)
#         bonus += (self.state[:, :-1] * 0.1).sum(axis=1, keepdims=True)
#         bonus += ((self.state[:, 0] > 0.5) * (self.state[:, 1] > 0.5) * 0.1).unsqueeze(1)
#         bonus += ((self.state[:, 0] > 0.9) * (self.state[:, 1] > 0.9) * 0.1).unsqueeze(1)
#         return bonus
#
#     def pick_action(self, eval_ep, state=None):
#         if state is None:
#             state = self.state
#         actions = []
#         for i, agent in enumerate(self.agents):
#             actions.append(agent.pick_action(eval_ep, state=state))
#         return torch.concatenate(actions)
#
#     def time_for_critic_and_actor_to_learn(self):
#         """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
#         actor and critic"""
#         return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
#                self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
#                    "update_every_n_steps"] == 0
#
#     def train(self):
#         self.run_n_episodes()
#
#     def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=False):
#         """Runs game to completion n times and then summarises results and saves model (if asked to)"""
#         if num_episodes is None: num_episodes = self.config.num_episodes_to_run
#         start = time.time()
#         self.eval()
#         with tqdm(total=num_episodes) as pbar:
#             while self.episode_number < num_episodes:
#                 self.reset_game()
#                 self.step()
#                 if self.episode_number % 200 == 0:
#                     self.save_model()
#                 if self.episode_number % self.config.test_frequency == 0:
#                     self.eval()
#                 pbar.update(1)
#         time_taken = time.time() - start
#         self.save_model()
#         return self.game_full_episode_scores, self.rolling_results, time_taken
#
#     def enough_experiences_to_learn_from(self):
#         """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
#         return len(self.memory) > self.hyperparameters["batch_size"]
#
#     def eval(self):
#         reward_list = []
#         objs_list = []
#         for i in range(self.config.episode_per_test):
#             self.reset_game()
#             episode_reward = 0
#             while not self.done:
#                 self.actions = self.pick_action(True)
#                 self.conduct_action(self.actions, eval=True)
#                 #objs_list.append(self.environment.tri_objs)
#                 episode_reward += self.reward.sum()
#                 self.state = self.next_state
#             reward_list.append(episode_reward)
#         #tri_objs = torch.stack(objs_list).mean(axis=0) - 0
#         self.config.writer.add_scalar('eval/reward', sum(reward_list) / self.config.episode_per_test,
#                                       self.episode_number)
#         self.config.writer.add_scalar('eval/reward_max', max(reward_list), self.episode_number)
#         self.config.writer.add_scalar('eval/reward_min', min(reward_list), self.episode_number)
#
#         # self.config.writer.add_scalar('eval/obj1', tri_objs[0], self.episode_number)
#         # self.config.writer.add_scalar('eval/obj2', tri_objs[1], self.episode_number)
#         # self.config.writer.add_scalar('eval/obj3', tri_objs[2], self.episode_number)
#
#     def save_model(self):
#         path = './model'
#         if os.path.exists(path) is False:
#             os.mkdir(path)
#         for i, a in enumerate(self.agents):
#             torch.save(a.actor_local.state_dict(), os.path.join(path, f'actor_{i}.pth'))
#
#     def load_model(self, path='./model'):
#         for i, a in enumerate(self.agents):
#             a.actor_local.load_state_dict(torch.load(os.path.join(path, f'actor_{i}.pth')))
#
#     def eval_with_graph(self):
#         self.reset_game()
#         states = []
#         actions = []
#         objs_list = []
#         while not self.done:
#             states.append(self.state)
#             # self.actions = torch.from_numpy(np.concatenate([np.ones(1), -0.75*np.ones(1), -0.75*np.ones(1),np.zeros(1)])).unsqueeze(0).repeat(8,1)
#             self.actions = self.pick_action(True)
#             actions.append(self.actions)
#             self.conduct_action(self.actions)
#             objs_list.append(self.environment.tri_objs)
#             self.state = self.next_state
#         states = np.array(torch.stack(states).cpu())
#         actions = np.array(torch.stack(actions).cpu())
#         tri_objs = torch.stack(objs_list).mean(axis=0)
#         print(tri_objs)
#         print(actions.mean(axis=1))
#         print(actions.mean(axis=0).mean(axis=0))
#         uav_postions = states[:, :, :]
#         plot_uav_positions(uav_postions, fig_name='masac')
import os
import time
import sys

import numpy as np
import torch
from tqdm import tqdm

from Base_Agent import Base_Agent
from SAC.SACAgent import SACAgent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from utilities.plot_uav_positions import plot_uav_positions

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class MASAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "MASAC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters
        self.environment = config.environment
        self.agents = [
            SACAgent(config, i, d['obs_space'].shape[0], d['action_space'].shape[0], agent_num=config.agent_num) for
            i, d in
            enumerate(config.agents_config_dict)]
        # --- 移除 self.do_evaluation_iterations ---
        #(self.device)  # 添加设备参数 # 修复：添加设备参数到重播缓冲区
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, self.device)

        # 在 self.episode_number = 0 下方添加
        self.total_episode_uav_energy = 0.0
        self.total_episode_charge = 0.0
        # --- 新增：用于存储最后一步的 info ---
        self.last_info = {}
        # --- 新增：学习步数统计 ---
        self.learning_steps_done = 0

        for a in self.agents:
            a.parent = self

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        # --- 新增：重置 episode 级别的累加器 ---
        self.total_episode_uav_energy = 0.0
        self.total_episode_charge = 0.0
        self.last_info = {}

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        reward_sum = 0
        while not self.done:
            if self.global_step_number < self.config.hyperparameters['min_steps_before_learning']:
                self.actions = np.random.random((len(self.agents), 4)) * 2 - 1
            else:
                self.actions = self.pick_action(False)

            # Convert actions to tensor if needed and conduct action
            if not isinstance(self.actions, torch.Tensor):
                self.actions = torch.tensor(self.actions, dtype=torch.float32)

            self.conduct_action(self.actions)
            reward_sum += self.reward.mean()
            if self.config.add_bonus:
                self.reward = self.reward + self.get_bonus(self.actions)

            if self.time_for_critic_and_actor_to_learn():
                for i in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.config.learning_updates = i
                    self.learn()
                self.learning_steps_done += 1

            mask = self.done
            # Ensure actions are numpy array before saving to replay buffer
            actions_to_save = self.actions.cpu().numpy() if isinstance(self.actions, torch.Tensor) else self.actions
            self.save_experience(
                experience=(self.state, actions_to_save, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1

        self.episode_number += 1

        # --- 修改：替换为新的 TensorBoard 日志记录 ---
        # 1. 记录总奖励
        self.config.writer.add_scalar('Reward/Train', reward_sum, self.episode_number)

        # 2. 记录其他指标 (来自 episode 的最后一步)
        if self.last_info:  # 确保 info 至少被赋值过一次
            # 2.1. 获取公平性指标 (来自最后一步)
            final_sense_fair = self.last_info.get('sense_fair', 0.0)
            final_charge_fair = self.last_info.get('charge_fair', 0.0)

            # 2.2. 获取地面设备平均剩余电量 (来自最后一步)
            final_gn_energy_list = self.last_info.get('gn_energy', [0.0])
            final_avg_gn_energy = np.mean(final_gn_energy_list)

            # 2.3. 记录到 writer
            self.config.writer.add_scalar('Metrics/Sensing_Fairness', final_sense_fair, self.episode_number)
            self.config.writer.add_scalar('Metrics/Charging_Fairness', final_charge_fair, self.episode_number)
            self.config.writer.add_scalar('Metrics/Avg_GN_Energy', final_avg_gn_energy, self.episode_number)

        # 3. 记录累加的能耗和充电
        self.config.writer.add_scalar('Metrics/Total_UAV_Energy', self.total_episode_uav_energy, self.episode_number)
        self.config.writer.add_scalar('Metrics/Total_GN_Charge_mJ', self.total_episode_charge, self.episode_number)
        # --- 日志记录修改结束 ---

    def produce_actions(self, states):
        # 验证下是否需要采样
        all_actor_acs = []
        for i, a in enumerate(self.agents):
            all_actor_acs.append(a.produce_action_and_action_info(states[:, i, :]))
        return all_actor_acs

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def sample_experiences(self):
        return self.memory.sample()

    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        for i in range(len(self.agents)):
            self.agents[i].learn(experiences=(state_batch, action_batch, reward_batch, next_state_batch, mask_batch))

    def conduct_action(self, actions, eval=False):
        # --- 修改：接收 info 字典 ---
        # 注意：eval 参数保留，以防 pick_action 依赖它（尽管此处 conduct_action 不再区分）
        self.next_state, self.reward, self.done, info = self.environment.step(actions.cpu().numpy())

        # 仅在训练时累加 (eval 默认为 False, train_SAC 中调用时也未传 True)
        if not eval:
            # --- 新增：累加能耗和充电 ---
            self.total_episode_uav_energy += info.get('step_uav_energy', 0.0)
            self.total_episode_charge += info.get('step_total_charge', 0.0)
            self.last_info = info  # 存储最后一步的 info

        self.total_episode_score_so_far += sum(self.reward)
        if self.hyperparameters["clip_rewards"]: self.reward = max(min(self.reward, 1.0), -1.0)

    def get_bonus(self, actions):
        bonus = torch.zeros_like(self.reward)
        bonus += (self.state[:, :-1] * 0.1).sum(axis=1, keepdims=True)
        bonus += ((self.state[:, 0] > 0.5) * (self.state[:, 1] > 0.5) * 0.1).unsqueeze(1)
        bonus += ((self.state[:, 0] > 0.9) * (self.state[:, 1] > 0.9) * 0.1).unsqueeze(1)
        return bonus

    def pick_action(self, eval_ep, state=None):
        if state is None:
            state = self.state
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.pick_action(eval_ep, state=state))
        return torch.concatenate(actions)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
            self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
                "update_every_n_steps"] == 0

    def train(self):
        self.run_n_episodes()

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=False):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        total_start_time = start

        # GPU利用率统计
        gpu_utilization = []
        episode_times = []

        # 初始化进度条
        if hasattr(self.config, 'performance_monitoring') and self.config.performance_monitoring:
            print(f"开始训练 {num_episodes} 个episode...")
            print(f"GPU启用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"当前使用GPU: {torch.cuda.current_device()}")

        # --- 移除 self.eval() 调用 ---
        with tqdm(total=num_episodes, desc="训练进度") as pbar:
            while self.episode_number < num_episodes:
                episode_start = time.time()
                self.reset_game()

                # 记录GPU内存使用
                if (hasattr(self.config, 'log_memory_usage') and self.config.log_memory_usage and
                    torch.cuda.is_available()):
                    start_memory = torch.cuda.memory_allocated()

                self.step()

                episode_time = time.time() - episode_start
                episode_times.append(episode_time)

                # 记录GPU内存变化
                if (hasattr(self.config, 'log_memory_usage') and self.config.log_memory_usage and
                    torch.cuda.is_available()):
                    end_memory = torch.cuda.memory_allocated()
                    memory_diff = (end_memory - start_memory) / 1024 / 1024  # MB
                    self.config.writer.add_scalar('Performance/GPU_Memory_Change_MB', memory_diff, self.episode_number)

                # 记录episode训练时间
                self.config.writer.add_scalar('Performance/Episode_Time_Seconds', episode_time, self.episode_number)

                # 定期保存模型
                if self.episode_number % 200 == 0:
                    self.save_model()

                # --- 详细的性能日志 ---
                if self.episode_number % 50 == 0 and self.episode_number > 0:
                    avg_time = np.mean(episode_times[-50:])
                    total_time = time.time() - total_start_time

                    print(f"\n[性能监控] Episode {self.episode_number}:")
                    print(f"  - 最近50个episode平均时间: {avg_time:.2f}秒")
                    print(f"  - 累计训练时间: {total_time/60:.1f}分钟")
                    print(f"  - 平均奖励: {self.total_episode_score_so_far:.3f}")

                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                        print(f"  - GPU内存使用: {gpu_memory:.1f} MB (已分配) / {gpu_memory_reserved:.1f} MB (已预留)")

                        # 添加TensorBoard监控
                        self.config.writer.add_scalar('Performance/GPU_Memory_Used_MB', gpu_memory, self.episode_number)
                        self.config.writer.add_scalar('Performance/GPU_Memory_Reserved_MB', gpu_memory_reserved, self.episode_number)

                    # 记录累计奖励和训练速度
                    self.config.writer.add_scalar('Performance/Average_Episode_Time', avg_time, self.episode_number)
                    self.config.writer.add_scalar('Performance/Total_Training_Time_Minutes', total_time/60, self.episode_number)

                # 渲染轨迹 (减少频率以优化性能)
                if self.episode_number % 1000 == 0 and self.episode_number > 0:  # 改为每1000个episode
                    print(f"[Render] Episode {self.episode_number} - Reward: {self.total_episode_score_so_far:.3f}")

                    try:
                        log_dir = self.config.writer.log_dir
                        run_id = os.path.basename(log_dir.strip('/'))
                        if not run_id:
                            run_id = "sac_default"
                        render_save_dir = f"./trajectory_plots/{run_id}"
                    except Exception as e:
                        print(f"Warning: Could not parse log_dir for render path. Error: {e}")
                        render_save_dir = "./trajectory_plots/sac_default"

                    os.makedirs(render_save_dir, exist_ok=True)
                    print(f"[Render] Saving trajectory to {render_save_dir}")
                    self.environment.render(save_dir=render_save_dir, episode_index=self.episode_number)

                pbar.set_postfix({
                    'Episode': self.episode_number,
                    'Reward': f'{self.total_episode_score_so_far:.2f}',
                    'Time': f'{episode_time:.1f}s'
                })
                pbar.update(1)

        total_time = time.time() - total_start_time
        self.save_model()

        print(f"\n训练完成!")
        print(f"总耗时: {total_time/60:.1f}分钟")
        print(f"总episode数: {self.episode_number}")
        print(f"平均每episode时间: {np.mean(episode_times):.2f}秒")

        return self.game_full_episode_scores, self.rolling_results, total_time

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    # --- 移除 eval(self) 方法 ---

    def save_model(self):
        path = './model'
        if os.path.exists(path) is False:
            os.mkdir(path)
        for i, a in enumerate(self.agents):
            torch.save(a.actor_local.state_dict(), os.path.join(path, f'actor_{i}.pth'))

    def load_model(self, path='./model'):
        for i, a in enumerate(self.agents):
            a.actor_local.load_state_dict(torch.load(os.path.join(path, f'actor_{i}.pth')))

    # --- 移除 eval_with_graph(self) 方法 ---