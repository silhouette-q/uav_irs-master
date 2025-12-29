import numpy as np

from utilities.RunningMeanStd import RunningMeanStd


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + np.array(x)
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x.float()

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)