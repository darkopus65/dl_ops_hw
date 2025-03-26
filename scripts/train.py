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
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    env = instantiate(config.environment)

    agent = instantiate(config.agent)

    raplay_buffer = instantiate(config.raplay_buffer)

    # logger.info(model)

    # get function handles of loss and metrics
    # loss_function = instantiate(config.loss_function).to(device)
    # metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer)
    # lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = BaseTrainer(
        agent=agent,
        env=env,
        memory=raplay_buffer,
        optimizer=optimizer,
        config=config,
        device=device,
        logger=logger,
    )

    trainer.train()


if __name__ == "__main__":
    main()