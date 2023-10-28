import pyrootutils
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig
from lightning_vocoders.models.hifigan.lightning_module import HiFiGANLightningModule
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import seed_everything

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_name="config", config_path="../config")
def main(cfg: DictConfig):
    seed_everything(1234)
    if cfg.train.ckpt_path is not None:
        lightning_module = hydra.utils.instantiate(cfg.model.lightning_module,cfg).load_from_checkpoint(cfg.train.ckpt_path,cfg=cfg)
    else:
        lightning_module = hydra.utils.instantiate(cfg.model.lightning_module, cfg)
    if cfg.compile:
        lightning_module = torch.compile(lightning_module, dynamic=True)
    callbacks = [LearningRateMonitor(logging_interval="step")]
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, cfg)
    loggers = [hydra.utils.instantiate(logger) for logger in cfg.train.loggers]
    trainer = hydra.utils.instantiate(
        cfg.train.trainer, logger=loggers, callbacks=callbacks
    )
    trainer.fit(lightning_module, datamodule)


if __name__ == "__main__":
    main()
