import random
from copy import copy, deepcopy

import numpy as np
import torch


def cal_ex_reward(l_s, s, a):
    l_s = torch.stack([l_s[i,i*3:i*3+3] for i in range(l_s.shape[0])])
    s = torch.stack([s[i,i*3:i*3+3] for i in range(l_s.shape[0])])
    
    beta1 = -0.5
    beta2 = -0.5
    beta3 = 0.2

    # 获取输入张量的设备，而不是硬编码.cuda()
    device = l_s.device
    bound = torch.Tensor([0.85, 0.85]).to(device)

    r1 = beta1 * torch.norm((bound - s[:, :-1]), dim=1)
    r2 = beta2 * torch.norm(
        (bound - s[:, :-1]) / torch.norm((bound - s[:, :-1])) - (s[:, :-1] - l_s[:, :-1]) / torch.norm(
            s[:, :-1] - l_s[:, :-1]), dim=1)
    r3 = beta3 * (a[:, 0] - 0.9)
    r4 = -5 * abs(a[:,1])
    return r1 + r2 + r3 + r4


def add_gravity_exploration_noise(action, ep_i_sum):
    action_noised = deepcopy(action)
    shape = (1,)

    # 获取输入张量的设备
    device = action.device
    # control speed  - 使用与输入张量相同的设备
    action_noised[:, 1] = torch.normal(mean=-0.015, std=0.1, size=(action.shape[0],)).to(device) \
                          * (1 - ep_i_sum) + action[:, 1] * ep_i_sum
    action_noised = torch.clip(action_noised, -1, 1)

    return action_noised
