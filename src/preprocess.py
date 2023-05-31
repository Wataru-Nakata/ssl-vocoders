import pyrootutils
import hydra
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from lightning_vocoders.preprocessor.preprocessor import Preprocessor


@hydra.main(version_base="1.3", config_name="config", config_path="../config")
def main(cfg: DictConfig):
    preprocssor = Preprocessor(cfg=cfg)
    preprocssor.build_from_path()


if __name__ == "__main__":
    main()
