import os
import time

import numpy as np
import torch
from tqdm import tqdm

from Base_Agent import Base_Agent
from SAL.SACAgent import SACAgent
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
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        for a in self.agents:
            a.parent = self

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        reward_sum = 0
        while not self.done:
            if self.global_step_number < self.config.hyperparameters['min_steps_before_learning']:
                self.actions = np.random.random((len(self.agents), 4)) * 2 - 1
            else:
                self.actions = self.pick_action(False)
            self.conduct_action(self.actions)
            reward_sum += self.reward
            if self.config.add_bonus:
                self.reward = self.reward + self.get_bonus(self.actions)
            if self.time_for_critic_and_actor_to_learn():
                for i in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.config.learning_updates = i
                    self.learn()
            mask = self.done
            self.save_experience(
                experience=(self.state, self.actions, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1
        self.config.writer.add_scalar('immediate_reward', reward_sum.sum(), self.episode_number)

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
        self.next_state, self.reward, self.done, _ = self.environment.step(actions.cpu().numpy())
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
        self.eval()
        with tqdm(total=num_episodes) as pbar:
            while self.episode_number < num_episodes:
                if self.episode_number % 50 == 0:
                    self.save_model()
                self.reset_game()
                self.step()
                if self.episode_number % self.config.test_frequency == 0:
                    self.eval()
                pbar.update(1)
        time_taken = time.time() - start
        self.save_model()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    def eval(self):
        reward_list = []
        objs_list = []
        for i in range(self.config.episode_per_test):
            self.reset_game()
            episode_reward = 0
            while not self.done:
                self.actions = self.pick_action(True)
                self.conduct_action(self.actions, eval=True)
                objs_list.append(self.environment.tri_objs)
                episode_reward += self.reward.sum()
                self.state = self.next_state
            reward_list.append(episode_reward)
        tri_objs = torch.stack(objs_list).mean(axis=0) - 0
        self.config.writer.add_scalar('eval/reward', sum(reward_list) / self.config.episode_per_test,
                                      self.episode_number)
        self.config.writer.add_scalar('eval/reward_max', max(reward_list), self.episode_number)
        self.config.writer.add_scalar('eval/reward_min', min(reward_list), self.episode_number)

        self.config.writer.add_scalar('eval/obj1', tri_objs[0], self.episode_number)
        self.config.writer.add_scalar('eval/obj2', tri_objs[1], self.episode_number)
        self.config.writer.add_scalar('eval/obj3', tri_objs[2], self.episode_number)

    def save_model(self):
        path = './SAL/model'
        if os.path.exists(path) is False:
            os.mkdir(path)
        for i, a in enumerate(self.agents):
            torch.save(a.actor_local.state_dict(), os.path.join(path, f'actor_{i}.pth'))

    def load_model(self, path='./SAL/model'):
        for i, a in enumerate(self.agents):
            a.actor_local.load_state_dict(torch.load(os.path.join(path, f'actor_{i}.pth')))

    def eval_with_graph(self):
        self.reset_game()
        states = []
        actions = []
        objs_list = []
        while not self.done:
            states.append(self.state)
            # self.actions = torch.from_numpy(np.concatenate([np.ones(1), -0.75*np.ones(1), -0.75*np.ones(1),np.zeros(1)])).unsqueeze(0).repeat(8,1)
            self.actions = self.pick_action(True)
            actions.append(self.actions)
            self.conduct_action(self.actions)
            objs_list.append(self.environment.tri_objs)
            self.state = self.next_state
        states = np.array(torch.stack(states).cpu())
        actions = np.array(torch.stack(actions).cpu())
        tri_objs = torch.stack(objs_list).mean(axis=0)
        print(tri_objs)
        print(actions.mean(axis=1))
        print(actions.mean(axis=0).mean(axis=0))
        uav_postions = states[:, :, :]
        plot_uav_positions(uav_postions, fig_name='masac')
