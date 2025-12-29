from collections import namedtuple, deque
import random
import torch
import numpy as np


class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed, device=None):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        # --- 开始修改 ---
        self.device = device if device is not None else torch.device("cpu")
        # --- 修改结束 ---

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = (states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        # states = torch.stack([experience[0] for experience in experiences])
        # actions = torch.stack([experience[1] for experience in experiences])
        # rewards = torch.stack([experience[2] for experience in experiences])
        # next_states = torch.stack([experience[3] for experience in experiences])
        # dones = None
        # --- 开始修改 (纠正版本) ---
        # np.array 会将 (N_Agents, N_Features) 的列表正确堆叠为 (Batch, N_Agents, N_Features)
        states = torch.from_numpy(np.stack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.stack([e[1] for e in experiences if e is not None])).float().to(self.device)

        # rewards 和 dones 已经是 (N_Agents,) 或 标量, 使用 vstack 变为 (Batch, N) 或 (Batch, 1) 是正确的
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        # --- 修改结束 ---
        return states, actions, rewards, next_states, dones

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
