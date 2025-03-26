import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

class BaseEnvironment(object):
    def __init__(self, env_name, screen_width):
        self.env_name = env_name
        self.env = gym.make(self.env_name, render_mode="rgb_array").unwrapped
        self.x_threshold = self.env.x_threshold
        self.state = self.env.state
        self.screen_width = screen_width

    def render(self):
        return self.env.render()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        _, reward, done, _, _ = self.env.step(action)
        return reward, done

    def close(self):
        self.env.close()

    def get_screen(self):
        screen = self.env.render().transpose(
            (2, 0, 1))  # transpose into torch order (CHW)
        # Strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location()
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(self.device)

    def get_cart_location(self):
        world_width = self.env.x_threshold * 2
        scale = self.screen_width / world_width
        return int(self.env.state[0] * scale + self.screen_width / 2.0)  # MIDDLE OF CART

    def resize(self, img):
        T.Compose([T.ToPILImage(),
        T.Resize(40, interpolation=Image.BICUBIC),
        T.ToTensor()])