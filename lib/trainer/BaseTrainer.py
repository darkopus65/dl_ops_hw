from abc import abstractmethod
from itertools import count
import torch
import torch.nn.functional as F
from lib.logger import plot_durations
from lib.utils.data_types import Transition
from lib.utils.io_utils import ROOT_PATH


class BaseTrainer:
    def __init__(self, agent, env, memory, optimizer, config, device, logger, writer, n_epochs=None):
        self.is_train = True
        self.agent = agent
        self.env = env
        self.config = config

        self.optimizer = optimizer

        self.device = device
        self.logger = logger
        self.writer = writer
        self.memory = memory
        self.BATCH_SIZE = self.config.trainer.batch_size
        self.TARGET_UPDATE = self.config.trainer.target_update
        self.GAMMA = self.config.trainer.gamma

        self.n_epochs = n_epochs
        self._last_epoch = 0
        self.episode_durations = []

        self.checkpoint_dir = (
                ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch)
            raise e

    def _train_process(self):
        for i_episode in range(self.n_epochs):
            self.writer.set_step(i_episode)
            self.writer.add_scalar("epoch", i_episode)
            self._last_epoch = i_episode
            self.env.reset()
            last_screen = self.env.get_screen()
            current_screen = self.env.get_screen()
            state = current_screen - last_screen
            for t in count():
                action = self.agent.select_action(state)
                reward, done = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                last_screen = current_screen
                current_screen = self.env.get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model(i_episode)
                if done:
                    self.episode_durations.append(t + 1)
                    self.writer.add_scalar("durations", self.episode_durations[-1])
                    plot_durations(self.episode_durations)
                    break

            if i_episode % self.TARGET_UPDATE == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
                self._save_checkpoint(i_episode)

    def optimize_model(self, episode):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.agent.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.agent.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.logger.debug(
            "Train Epoch: {}, Loss: {:.6f}".format(
                episode,loss
            )
        )
        self.writer.add_scalar(
            "Loss", loss
        )


        self.optimizer.zero_grad()
        loss.backward()
        for param in self.agent.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    def _save_checkpoint(self, epoch):
        env = type(self.env).__name__
        state = {
            "env": env,
            "agent": self.agent,
            "policy_net": self.agent.policy_net.state_dict(),
            "target_net": self.agent.target_net.state_dict(),
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "buffer": self.memory,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        best_path = str(self.checkpoint_dir / "model_best.pth")
        torch.save(state, filename)
        torch.save(state, best_path)
        if self.config.writer.log_checkpoints:
            self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
        self.logger.info(f"Saving checkpoint: {filename} ...")
    def _from_pretrained(self, pretrained_path):
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device, weights_only=False)


        self.agent.policy_net.load_state_dict(checkpoint["policy_net"])
        self.agent.target_net.load_state_dict(checkpoint["target_net"])
