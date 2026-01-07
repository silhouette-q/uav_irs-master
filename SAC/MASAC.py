import os
import time
import numpy as np
import torch
from tqdm import tqdm

from Base_Agent import Base_Agent
from SAC.SACAgent import SACAgent
from utilities.data_structures.Replay_Buffer import Replay_Buffer


class MASAC(Base_Agent):
    """Multi-Agent Soft Actor-Critic algorithm."""
    agent_name = "MASAC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.agent_num = config.agent_num

        assert self.action_types == "CONTINUOUS", "Action types must be continuous."
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax."

        self.hyperparameters = config.hyperparameters
        self.environment = config.environment

        # --- 关键修复：在这里定义 self.action_size ---
        self.action_size = config.agents_config_dict[0]['action_space'].shape[0]

        self.agents = [
            SACAgent(config, i, d['obs_space'].shape[0], d['action_space'].shape[0], agent_num=config.agent_num)
            for i, d in enumerate(config.agents_config_dict)
        ]

        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, self.device)

        self.last_info = {}
        self.learning_steps_done = 0

        for a in self.agents:
            a.parent = self

    def reset_game(self):
        """Resets the game information for a new episode."""
        Base_Agent.reset_game(self)
        self.last_info = {}
        self.state = self.environment.reset()

    def step(self):
        """Runs a single episode, saving experience and learning."""
        episode_reward_sum = 0

        while not self.done:
            if self.global_step_number < self.hyperparameters['min_steps_before_learning']:
                # --- 关键修复：使用 self.action_size 生成正确维度的随机动作 ---
                self.actions = np.random.uniform(-1.0, 1.0, size=(self.agent_num, self.action_size))
            else:
                self.actions = self.pick_action(eval_ep=False)

            self.conduct_action(self.actions)

            episode_reward_sum += self.reward.sum()

            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
                self.learning_steps_done += 1

            actions_to_save = self.actions.cpu().numpy() if isinstance(self.actions, torch.Tensor) else self.actions
            self.save_experience(experience=(self.state, actions_to_save, self.reward, self.next_state, self.done))

            self.state = self.next_state
            self.global_step_number += 1

        self.episode_number += 1
        self.log_episode_results(episode_reward_sum)

    def log_episode_results(self, episode_reward_sum):
        """Logs results of the completed episode to TensorBoard."""
        self.config.writer.add_scalar('Reward/Train_Episode_Total', episode_reward_sum, self.episode_number)

        if self.last_info:
            self.config.writer.add_scalar('Metrics/Total_Detections', self.last_info.get('total_detections', 0),
                                          self.episode_number)
            self.config.writer.add_scalar('Metrics/New_Detections', self.last_info.get('new_detections', 0),
                                          self.episode_number)
            self.config.writer.add_scalar('Metrics/Total_Charge_mJ', self.last_info.get('total_charge_mj', 0.0),
                                          self.episode_number)
            self.config.writer.add_scalar('Metrics/Avg_UAV_Energy_Left', self.last_info.get('avg_uav_energy_left', 0.0),
                                          self.episode_number)

    def conduct_action(self, actions):
        """Conducts an action in the environment."""
        actions_np = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        self.next_state, self.reward, self.done, self.last_info = self.environment.step(actions_np)
        self.total_episode_score_so_far += self.reward.sum()

    def pick_action(self, eval_ep=False, state=None):
        """Picks an action for all agents."""
        if state is None: state = self.state
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.device)

        actions = []
        for i, agent in enumerate(self.agents):
            agent_state = state[i, :]
            actions.append(agent.pick_action(eval_ep, state=agent_state))

        return torch.cat(actions, dim=0).reshape(self.agent_num, -1)

    def learn(self):
        """Samples a batch and triggers learning for all agents."""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        for i in range(len(self.agents)):
            self.agents[i].learn(experiences=(state_batch, action_batch, reward_batch, next_state_batch, mask_batch))

    def run_n_episodes(self, num_episodes=None):
        """Runs the training loop."""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run

        with tqdm(total=num_episodes, desc="Training Progress") as pbar:
            while self.episode_number < num_episodes:
                self.reset_game()
                self.step()

                pbar.set_postfix({
                    'Episode': self.episode_number,
                    'Total Reward': f'{self.total_episode_score_so_far:.2f}',
                    'Steps': self.global_step_number
                })
                pbar.update(1)

                if self.episode_number > 0 and self.episode_number % 200 == 0 and self.config.save_model:
                    self.save_model()

                if self.episode_number > 0 and self.episode_number % 1000 == 0:
                    print(f"\n[Render] Rendering trajectory for episode {self.episode_number}...")
                    try:
                        render_save_dir = os.path.join(self.config.writer.log_dir, "trajectories")
                        self.environment.render(save_dir=render_save_dir, episode_index=self.episode_number)
                    except Exception as e:
                        print(f"Warning: Could not render trajectory. Error: {e}")

        if self.config.save_model:
            self.save_model()

    def save_experience(self, memory=None, experience=None):
        if memory is None: memory = self.memory
        memory.add_experience(*experience)

    def sample_experiences(self):
        return self.memory.sample()

    def time_for_critic_and_actor_to_learn(self):
        return (self.global_step_number > self.hyperparameters["min_steps_before_learning"] and
                len(self.memory) > self.hyperparameters["batch_size"] and
                self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0)

    def train(self):
        self.run_n_episodes()

    def save_model(self):
        path = os.path.join(self.config.writer.log_dir, "model")
        os.makedirs(path, exist_ok=True)
        for i, a in enumerate(self.agents):
            torch.save(a.actor_local.state_dict(), os.path.join(path, f'actor_{i}.pth'))
        print(f"\nModel saved to {path}")

    def load_model(self, path='./model'):
        for i, a in enumerate(self.agents):
            a.actor_local.load_state_dict(torch.load(os.path.join(path, f'actor_{i}.pth')))
