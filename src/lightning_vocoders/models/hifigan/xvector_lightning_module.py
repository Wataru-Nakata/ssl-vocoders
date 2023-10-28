from omegaconf import DictConfig
from .lightning_module import HiFiGANLightningModule
from .generator_xvector import GeneratorWithXvector

class HiFiGANXvectorLightningModule(HiFiGANLightningModule,object):
    def __init__(self, cfg: DictConfig) -> None:
        HiFiGANLightningModule.__init__(self,cfg)
        self.generator = GeneratorWithXvector(cfg.model.generator)
    
    def generator_forward(self, batch):
        wav_generator_out = self.generator(batch["input_feature"],batch['xvector'])
        return wav_generator_out
