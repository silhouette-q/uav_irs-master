from Base_Agent import Base_Agent
from utilities.Gravity_Exploration_Noise import add_gravity_exploration_noise
from utilities.data_structures.Replay_Buffer import Replay_Buffer
import torch
import torch.nn.functional as F
import numpy as np

from utilities.exploration_strategies.Gaussian_Exploration import Gaussian_Exploration

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class DDPGAgent(Base_Agent):
    agent_name = "DDPG"

    def __init__(self, config, agent_id, state_size, action_size, agent_num=8):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters
        self.config.action_size = action_size
        if state_size is not None:
            self.state_size = state_size
        if action_size is not None:
            self.action_size = action_size

        self.agent_id = agent_id
        self.agent_num = agent_num
        self.critic_local = self.create_NN(input_dim=self.state_size + (self.action_size) * agent_num, output_dim=1,
                                           key_to_use="Critic")
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size + (self.action_size) * agent_num, output_dim=1,
                                            key_to_use="Critic")
        self.action_space = config.agents_config_dict[self.agent_id]['action_space']
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                          key_to_use="Actor")
        self.actor_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                           key_to_use="Actor")
        Base_Agent.copy_model_over(self.actor_local, self.actor_target)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)

        self.exploration_strategy = Gaussian_Exploration(self.config)

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def pick_action(self, state=None, eval_ep=False):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if state is None:
            state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state)
        self.actor_local.train()

        if not eval_ep:
            if self.config.hyperparameters['add_gravity_noise']:
                action = add_gravity_exploration_noise(state, action)
            noise_std = self.hyperparameters["action_noise_std"]
            # * max(self.config.num_episodes_to_run - self.episode_number, 0.1) / self.config.num_episodes_to_run
            action = torch.clip(torch.distributions.normal.Normal(0, noise_std).sample(sample_shape=action.shape).cuda(), -0.5, 0.5) + action
        action = torch.clip(action, -1, 1)
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        self.critic_learn(states, actions, rewards, next_states, None)
        self.actor_learn(states, actions)

    def soft_update(self):
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target,
                                           self.hyperparameters["Actor"]["tau"])

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for the critic"""
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = F.mse_loss(critic_expected, critic_targets)
        if self.agent_id == 0 and self.config.learning_updates == self.hyperparameters[
            "learning_updates_per_learning_session"] - 1:
            self.config.writer.add_scalar('train_uav/critic1_loss', loss.item(), self.parent.global_step_number)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.parent.all_actions_target(next_states)
            critic_targets_next = self.critic_target(torch.cat((next_states[:,self.agent_id,:].reshape(self.config.hyperparameters['batch_size'], -1), actions_next), 1))
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards[:, self.agent_id] + (self.hyperparameters["discount_rate"] * critic_targets_next)
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.critic_local(
            torch.cat((states[:,self.agent_id,:].reshape(self.config.hyperparameters['batch_size'], -1), actions.reshape(self.config.hyperparameters['batch_size'], -1)), 1))
        return critic_expected

    def actor_learn(self, states, actions):
        """Runs a learning iteration for the actor"""
        actor_loss = self.calculate_actor_loss(states, actions)
        if self.agent_id == 0 and self.config.learning_updates == self.hyperparameters[
            "learning_updates_per_learning_session"] - 1:
            self.config.writer.add_scalar('train_uav/actor_loss', actor_loss.item(), self.parent.global_step_number)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])

    def calculate_actor_loss(self, states, actions):
        """Calculates the loss for the actor"""
        actions_input = []
        for i in range(self.agent_num):
            if i == self.agent_id:
                actions_input.append(self.actor_local(states[:, self.agent_id]))
            else:
                actions_input.append(actions[:, i, :])

        actions_input = torch.stack(actions_input, axis=1)

        actor_loss = -self.critic_local(torch.cat((states[:,self.agent_id,:].reshape(self.config.hyperparameters['batch_size'], -1),
                                                   actions_input.reshape(self.config.hyperparameters['batch_size'], -1)), 1)).mean()

        return actor_loss

    def print_summary_of_latest_evaluation_episode(self):
        pass
