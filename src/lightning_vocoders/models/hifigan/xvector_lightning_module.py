from omegaconf import DictConfig
from .lightning_module import HiFiGANLightningModule

class HiFiGANXvectorLightningModule(HiFiGANLightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.generator = GeneratorXvector(cfg.model.generator)
    
    def generator_forward(self, batch):
        wav_generator_out = self.generator(batch["input_feature"],batch['xvector'])
        return wav_generator_out