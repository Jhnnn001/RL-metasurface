"""Minimal discrete DQN over period, slit width, and metal thickness."""
import random
import numpy as np
import torch
from torch import nn


def valid_actions(state, shape=(5, 5, 5)):
    s = tuple(state)
    return np.array([s[a // 2] > 0 if a % 2 == 0 else s[a // 2] < shape[a // 2] - 1
                     for a in range(6)], dtype=bool)


def move(state, action, shape=(5, 5, 5)):
    if not 0 <= action < 6 or not valid_actions(state, shape)[action]:
        raise ValueError("action leaves the design grid")
    s = list(state); axis, step = action // 2, -1 if action % 2 == 0 else 1
    s[axis] += step
    return tuple(s)


def bellman_target(reward, next_max, done, gamma=0.99):
    return reward + gamma * next_max * (~done).float()


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 6))
    def forward(self, x): return self.net(x)


def demo(seed=0, episodes=8):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    net = QNet(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    target = (4, 4, 4)
    for _ in range(episodes):
        s = (2, 2, 2)
        for _ in range(12):
            actions = np.flatnonzero(valid_actions(s))
            a = int(random.choice(actions)); ns = move(s, a)
            r = 1.0 if ns == target else -0.01
            x = torch.tensor([s], dtype=torch.float32); y = net(x).clone().detach()
            y[0, a] = r + 0.99 * net(torch.tensor([ns], dtype=torch.float32)).max().detach()
            loss = nn.functional.mse_loss(net(x), y); opt.zero_grad(); loss.backward(); opt.step(); s = ns
    return net
