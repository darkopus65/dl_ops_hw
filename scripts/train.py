import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# from lib.datasets.data_utils import get_dataloaders
from lib.trainer.BaseTrainer import BaseTrainer
from lib.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="configs", config_name="baseline")
def main(config):
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    env = instantiate(config.environment)
    env.device = device

    policy_net = instantiate(config.model).to(device)
    logger.info(policy_net)
    target_net = instantiate(config.model).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    logger.info(target_net)

    rl_agent = instantiate(config.rl_agent)

    rl_agent.policy_net = policy_net
    rl_agent.target_net = target_net

    replayBuffer = instantiate(config.replayBuffer)

    optimizer = instantiate(config.optimizer, params=rl_agent.parameters())

    n_epochs = config.trainer.get("n_epochs")

    trainer = BaseTrainer(
        agent=rl_agent,
        env=env,
        memory=replayBuffer,
        optimizer=optimizer,
        config=config,
        device=device,
        logger=logger,
        writer=writer,
        n_epochs=n_epochs,
    )

    trainer.train()


if __name__ == "__main__":
    main()