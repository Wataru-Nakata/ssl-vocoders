import pyrootutils
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig
from lightning_vocoders.models.hifigan.lightning_module import HiFiGANLightningModule

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_name="config", config_path="../config")
def main(cfg: DictConfig):
    lightning_module = hydra.utils.instantiate(cfg.model.lightning_module, cfg)
    if cfg.compile:
        lightning_module = torch.compile(lightning_module)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, cfg)
    loggers = [hydra.utils.instantiate(logger) for logger in cfg.train.loggers]
    trainer = hydra.utils.instantiate(cfg.train.trainer, logger=loggers)
    trainer.fit(lightning_module, datamodule)


if __name__ == "__main__":
    main()
