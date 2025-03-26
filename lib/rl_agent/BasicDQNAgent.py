import math
import random

import torch


class BasicDQNAgent:
    def __init__(self, policy_net, target_net, EPS_END, EPS_START, EPS_DECAY):
        self.policy_net = policy_net
        self.target_net = target_net
        self.EPS_END = EPS_END
        self.EPS_START = EPS_START
        self.EPS_DECAY = EPS_DECAY
        self.device = None
        self.steps_done = 0


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

