import numpy as np
import torch
from matplotlib import pyplot as plt

from environment.env import UavIrsEnv

done = False
env = UavIrsEnv()
state = env.reset()
user_pos_list = []
evsdp_pos_list = []
while not done:
    actions = torch.from_numpy(np.concatenate([np.ones(1), -0.75 * np.ones(1), -0.75 * np.ones(1), np.zeros(1)])).unsqueeze(0).repeat(8, 1)
    _, _, done, _ = env.step(actions=actions.cpu().numpy())
    user_pos_list.append(env.user_pos)
    evsdp_pos_list.append(env.evsdp_pos)

user_pos_list = np.stack(user_pos_list)
evsdp_pos_list = np.stack(evsdp_pos_list)

plt.plot(user_pos_list[:, 0], user_pos_list[:, 1])
plt.plot(evsdp_pos_list[:, 0], evsdp_pos_list[:, 1])
plt.show()
