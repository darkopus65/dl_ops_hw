import warnings

import hydra
import torch
from hydra.utils import instantiate
from lib.utils.io_utils import ROOT_PATH
from lib.trainer.Inferencer import Inferencer

from lib.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    env = instantiate(config.environment)
    env.device = device

    policy_net = instantiate(config.model).to(device)
    target_net = instantiate(config.model).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    rl_agent = instantiate(config.rl_agent)

    rl_agent.policy_net = policy_net
    rl_agent.target_net = target_net


    inferencer = Inferencer(
        agent=rl_agent,
        env=env,
        config=config,
        device=device,
    )

    inferencer.run_inference()


if __name__ == "__main__":
    main()