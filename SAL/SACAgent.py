from Base_Agent import Base_Agent
from utilities.Gravity_Exploration_Noise import add_gravity_exploration_noise
from utilities.OU_Noise import OU_Noise
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class SACAgent(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config, agent_id, state_size, action_size, agent_num=8):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters

        if state_size is not None:
            self.state_size = state_size
        if action_size is not None:
            self.action_size = action_size

        self.agent_id = agent_id
        self.agent_num = agent_num
        self.critic_local = self.create_NN(input_dim=self.state_size + (self.action_size) * 1, output_dim=1,
                                           key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + (self.action_size) * 1, output_dim=1,
                                             key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size + (self.action_size) * 1, output_dim=1,
                                            key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + (self.action_size) * 1, output_dim=1,
                                              key_to_use="Critic")
        self.action_space = config.agents_config_dict[self.agent_id]['action_space']
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2,
                                          key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.log(torch.Tensor([0.2])).to(self.device).requires_grad_(True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["alpha_learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        if self.add_extra_noise: self.noise.reset()

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""

        if state is None: state = self.state
        if eval_ep:
            if self.config.deterministic_eval:
                action = self.actor_pick_action(state=state[self.agent_id, :], eval=True)
            else:
                action = self.actor_pick_action(state=state[self.agent_id, :])
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.action_space.sample()
        else:
            action = self.actor_pick_action(state=state[self.agent_id, :])

        if not eval_ep:
            self.global_step_number += 1
            if self.add_extra_noise:
                action += self.noise.sample()
            if self.config.add_gravity_noise:
                if np.random.random(1) > self.parent.episode_number / self.config.num_episodes_to_run:
                    action = add_gravity_exploration_noise(state[self.agent_id, :].unsqueeze(0), action)
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None: state = self.state
        if len(state.shape) == 1: state = state.unsqueeze(0)
        with torch.no_grad():
            if eval is False:
                action, _, _ = self.produce_action_and_action_info(state)
            else:
                _, z, action = self.produce_action_and_action_info(state)
        return action

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
            self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
                "update_every_n_steps"] == 0

    def learn(self, experiences):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = experiences
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                          mask_batch)
        self.update_critic_parameters(qf1_loss, qf2_loss)

        policy_loss, log_pi = self.calculate_actor_loss(state_batch, action_batch)
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_actor_parameters(policy_loss, alpha_loss)

        if self.need_log():
            self.config.writer.add_scalar('train_uav/actor_loss', policy_loss.item(), self.parent.global_step_number)
            self.config.writer.add_scalar('train_uav/critic1_loss', qf1_loss.item(), self.parent.global_step_number)
            self.config.writer.add_scalar('train_uav/critic2_loss', qf2_loss.item(), self.parent.global_step_number)
            if self.automatic_entropy_tuning:
                self.config.writer.add_scalar('train_uav/alpha_loss', alpha_loss.item(), self.parent.global_step_number)

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            all_agents_actions = self.parent.produce_actions(next_state_batch)
            next_state_action = torch.concatenate([a[0] for i, a in enumerate(all_agents_actions)], 1)
            next_state_log_pi = all_agents_actions[self.agent_id][1]

            critic_tar_input_batch = torch.cat([next_state_batch[:,self.agent_id,:].reshape(next_state_batch.shape[0], -1),
                                                all_agents_actions[self.agent_id][0]], axis=1)
            qf1_next_target = self.critic_target(critic_tar_input_batch)
            qf2_next_target = self.critic_target_2(critic_tar_input_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch[:, self.agent_id] + self.hyperparameters["discount_rate"] * (
                min_qf_next_target)

        critic_input_batch = torch.cat(
            [state_batch[:,self.agent_id,:].reshape(state_batch.shape[0], -1),
             action_batch[:,self.agent_id,:].reshape(state_batch.shape[0], -1)], axis=1)
        qf1 = self.critic_local(critic_input_batch)
        qf2 = self.critic_local_2(critic_input_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch, action_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        actions_input = []
        for i in range(self.agent_num):
            if i == self.agent_id:
                actions_curr_agent = self.produce_action_and_action_info(state_batch[:, self.agent_id])
                actions_input.append(actions_curr_agent[0])
                log_pi = actions_curr_agent[1]
            else:
                actions_input.append(action_batch[:, i, :])
        actions_input = torch.stack(actions_input, axis=1)
        critic_input_batch = torch.cat(
            [state_batch[:,self.agent_id,:].reshape(state_batch.shape[0], -1),
             actions_input[:,self.agent_id,:].reshape(actions_input.shape[0], -1)], axis=1)
        qf1_pi = self.critic_local(critic_input_batch)
        qf2_pi = self.critic_local_2(critic_input_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for both critics"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()
            if self.need_log():
                self.config.writer.add_scalar('train_uav/alpha', self.alpha, self.parent.global_step_number)

    def print_summary_of_latest_evaluation_episode(self):
        pass

    def need_log(self):
        if self.agent_id == 0 and self.config.learning_updates == self.hyperparameters[
            "learning_updates_per_learning_session"] - 1 and self.parent.global_step_number % self.config.log_interval == 0:
            return True
