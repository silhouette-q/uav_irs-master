import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from Base_Agent import Base_Agent

EPSILON = 1e-6


class SACAgent(Base_Agent):
    """Soft Actor-Critic agent for multi-agent settings."""
    agent_name = "SAC"

    def __init__(self, config, agent_id, state_size, action_size, agent_num):
        Base_Agent.__init__(self, config)
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.agent_num = agent_num
        self.hyperparameters = config.hyperparameters

        # Critic Networks
        # --- 关键修复：Critic 输入维度应为所有智能体状态和动作的总和 ---
        critic_input_dim = self.state_size * self.agent_num + self.action_size * self.agent_num

        self.critic_local = self.create_NN(input_dim=critic_input_dim, output_dim=1, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=critic_input_dim, output_dim=1, key_to_use="Critic")
        self.critic_target = self.create_NN(input_dim=critic_input_dim, output_dim=1, key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=critic_input_dim, output_dim=1, key_to_use="Critic")

        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=self.hyperparameters["Critic"]["learning_rate"],
                                     eps=1e-4)
        self.critic_optimizer_2 = Adam(self.critic_local_2.parameters(),
                                       lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)

        # Actor Network
        # --- 确认 Actor 输入输出维度正确 ---
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2,
                                          key_to_use="Actor")
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=self.hyperparameters["Actor"]["learning_rate"],
                                    eps=1e-4)

        # Entropy tuning
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor((self.action_size,))).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["alpha_learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

    def pick_action(self, eval_ep=False, state=None):
        """Picks an action from the actor network."""
        if state is None:
            raise ValueError("State cannot be None for pick_action")
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.device)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            if eval_ep and self.config.deterministic_eval:
                _, _, action = self.produce_action_and_action_info(state)
            else:
                action, _, _ = self.produce_action_and_action_info(state)
        return action

    def produce_action_and_action_info(self, state):
        """Produces an action, log probability, and mean action from the actor network."""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]

        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean)

    def learn(self, experiences):
        """Runs a learning iteration."""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = experiences

        # --- Critic Update ---
        with torch.no_grad():
            next_actions = []
            next_log_pis = []
            for i in range(self.agent_num):
                agent_next_state = next_state_batch[:, i, :]
                next_action, next_log_pi, _ = self.parent.agents[i].produce_action_and_action_info(agent_next_state)
                next_actions.append(next_action)
                next_log_pis.append(next_log_pi)

            next_actions_cat = torch.cat(next_actions, dim=1)
            next_log_pi_agent = next_log_pis[self.agent_id]

            critic_target_input = torch.cat(
                (next_state_batch.view(self.hyperparameters["batch_size"], -1), next_actions_cat), dim=1)

            qf1_next_target = self.critic_target(critic_target_input)
            qf2_next_target = self.critic_target_2(critic_target_input)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi_agent

            reward_for_agent = reward_batch[:, self.agent_id].unsqueeze(1)
            next_q_value = reward_for_agent + (1.0 - mask_batch) * self.hyperparameters[
                "discount_rate"] * min_qf_next_target

        critic_input = torch.cat((state_batch.view(self.hyperparameters["batch_size"], -1),
                                  action_batch.view(self.hyperparameters["batch_size"], -1)), dim=1)
        qf1 = self.critic_local(critic_input)
        qf2 = self.critic_local_2(critic_input)

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        self.critic_optimizer.zero_grad()
        qf1_loss.backward()
        self.critic_optimizer.step()

        self.critic_optimizer_2.zero_grad()
        qf2_loss.backward()
        self.critic_optimizer_2.step()

        # --- Actor and Alpha Update (Delayed for stability) ---
        # Only update actor every N critic updates, common practice in SAC
        if self.parent.learning_steps_done % 1 == 0:
            actor_actions = []
            actor_log_pis = []
            for i in range(self.agent_num):
                agent_state = state_batch[:, i, :]
                if i == self.agent_id:
                    action, log_pi, _ = self.produce_action_and_action_info(agent_state)
                    actor_actions.append(action)
                    actor_log_pis.append(log_pi)
                else:
                    with torch.no_grad():
                        # Get actions from other agents' current policies for actor update
                        other_agent_action, _, _ = self.parent.agents[i].produce_action_and_action_info(agent_state)
                        actor_actions.append(other_agent_action)

            actor_actions_cat = torch.cat(actor_actions, dim=1)
            log_pi_agent = actor_log_pis[0]

            critic_actor_input = torch.cat(
                (state_batch.view(self.hyperparameters["batch_size"], -1), actor_actions_cat), dim=1)

            # Freeze critic gradients for this part
            for p in self.critic_local.parameters(): p.requires_grad = False
            for p in self.critic_local_2.parameters(): p.requires_grad = False

            qf1_pi = self.critic_local(critic_actor_input)
            qf2_pi = self.critic_local_2(critic_actor_input)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi_agent) - min_qf_pi).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze critic gradients
            for p in self.critic_local.parameters(): p.requires_grad = True
            for p in self.critic_local_2.parameters(): p.requires_grad = True

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi_agent + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = None

            # --- Soft Update Target Networks ---
            self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                               self.hyperparameters["Critic"]["tau"])
            self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                               self.hyperparameters["Critic"]["tau"])

            # --- Logging ---
            if self.agent_id == 0 and self.parent.global_step_number % self.config.log_interval == 0:
                self.config.writer.add_scalar(f'Loss/Actor', policy_loss.item(), self.parent.global_step_number)
                self.config.writer.add_scalar(f'Loss/Critic1', qf1_loss.item(), self.parent.global_step_number)
                self.config.writer.add_scalar(f'Loss/Critic2', qf2_loss.item(), self.parent.global_step_number)
                if alpha_loss is not None:
                    self.config.writer.add_scalar(f'Loss/Alpha', alpha_loss.item(), self.parent.global_step_number)
                self.config.writer.add_scalar(f'Values/Alpha', self.alpha.item(), self.parent.global_step_number)
