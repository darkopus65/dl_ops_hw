import math
import random

import torch


class BasicDQNAgent:
    def __init__(self, EPS_END, EPS_START, EPS_DECAY, IS_TRAIN):
        self.EPS_END = EPS_END
        self.EPS_START = EPS_START
        self.EPS_DECAY = EPS_DECAY
        self.device = None
        self.steps_done = 0
        self.policy_net = None
        self.target_net = None
        self.IS_TRAIN = IS_TRAIN


    def select_action(self, state):
        if self.IS_TRAIN is False:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def parameters(self):
        return self.policy_net.parameters()

