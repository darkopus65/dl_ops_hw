from lib.logger.utils import plot_env
from lib.trainer.BaseTrainer import BaseTrainer
from itertools import count

class Inferencer(BaseTrainer):
    def __init__(
            self, agent, env, config, device, skip_model_load=False
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.device = device

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        self.env.reset()
        last_screen = self.env.get_screen()
        current_screen = self.env.get_screen()
        state = current_screen - last_screen
        for t in count():
            plot_env(self.env.get_screen())
            action = self.agent.select_action(state)
            reward, done = self.env.step(action.item())

            last_screen = current_screen
            current_screen = self.env.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            state = next_state

            if done:
                break