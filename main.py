import hydra
from omegaconf import DictConfig
from run_experiment import run_experiment


@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """
    Run the Hydra-configured experiment.

    Args:
        cfg: Resolved experiment configuration.
    """
    run_experiment(cfg)


if __name__ == "__main__":
    main()
