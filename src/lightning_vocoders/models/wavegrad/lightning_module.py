from typing import Any
from lightning.pytorch import LightningModule, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT
import hydra
from .wavegrad import WaveGrad
import numpy as np
import torch

class WaveGradLightningModule(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = WaveGrad(**cfg.model.model_params)
        self.cfg = cfg
        self.criterion = torch.nn.L1Loss()
        self.num_steps = 1000
        self.save_hyperparameters()
    def setup(self, stage: str) -> None:
        self.beta = np.linspace(**self.cfg.model.noise_schedule)
        self.alpha = 1-self.beta
        self.alpha_cum = np.cumprod(self.alpha)
        noise_level = np.cumprod(1-self.beta)**0.5
        noise_level = np.concatenate([[1.0], noise_level], axis=0)
        self.noise_level = torch.tensor(noise_level, dtype=torch.float32,device=self.device)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        wav,input_feature = batch['resampled_speech.pth'], batch['input_feature']
        batch_size = wav.size(0)
        wav = wav[:,:int(input_feature.size(1)/50*22050)]

        s = torch.randint(1, self.num_steps + 1, [batch_size], device=self.device)
        l_a, l_b = self.noise_level.to(self.device)[s - 1], self.noise_level.to(self.device)[s]
        noise_scale = l_a + torch.rand(batch_size, device=self.device) * (l_b - l_a)
        noise_scale = noise_scale.unsqueeze(1)
        noise = torch.randn_like(wav)

        noisy_wav = noise_scale * wav + (1.0 - noise_scale**2)**0.5 * noise

        predicted = self.model(noisy_wav, input_feature.transpose(1,2),noise_scale.squeeze(1))
        loss = self.criterion(predicted, noise)
        self.log('train/loss',loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        wav,input_feature = batch['resampled_speech.pth'], batch['input_feature']
        batch_size = wav.size(0)
        wav = wav[:,:int(input_feature.size(1)/50*22050)]

        s = torch.randint(1, self.num_steps + 1, [batch_size], device=self.device)
        l_a, l_b = self.noise_level.to(self.device)[s - 1], self.noise_level.to(self.device)[s]
        noise_scale = l_a + torch.rand(batch_size, device=self.device) * (l_b - l_a)
        noise_scale = noise_scale.unsqueeze(1)
        noise = torch.randn_like(wav)

        noisy_wav = noise_scale * wav + (1.0 - noise_scale**2)**0.5 * noise

        print(noisy_wav.size(), wav.size())
        predicted = self.model(noisy_wav, input_feature.transpose(1,2),noise_scale.squeeze(1))
        loss = self.criterion(predicted, noise)
        self.log('val/loss',loss)
        if batch_idx < self.cfg.model.n_logging_wav_samples and self.global_rank == 0 and self.local_rank == 0:
            predicted_audio = self.predict(input_feature[0].unsqueeze(0), wav.size(1))
            self.log_audio(predicted_audio[0].cpu().numpy().astype(np.float32),name=f"generated/{batch['filenames'][0]}",sampling_rate=self.cfg.sample_rate)
        return loss
    def predict(self, input_feature,audio_length):
        audio = torch.randn((1, audio_length),device=self.device)
        noise_scale = torch.from_numpy(self.alpha_cum**0.5).float().unsqueeze(1).to(self.device)

        for n in range(len(self.alpha) - 1 , -1 ,-1):
            c1= 1/ (self.alpha[n] ** 0.5)

            c2 = (1- self.alpha[n]) / (1-self.alpha_cum[n]) ** 0.5
            audio = c1 * (audio - c2 * self.model(audio,input_feature.transpose(1,2),noise_scale[n]).squeeze(1))
            if n> 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - self.alpha_cum[n-1])/ (1.0 - self.alpha[n-1]) * self.beta[n]) ** 0.5
                audio += sigma*noise
            audio = torch.clamp(audio, -1.0, 1.0)
        return audio
    def configure_optimizers(self) -> Any:
        return hydra.utils.instantiate(self.cfg.model.optim, params=self.parameters())
    
    def log_audio(self, audio, name, sampling_rate):
        for logger in self.loggers:
            match type(logger):
                case loggers.WandbLogger:
                    import wandb

                    wandb.log(
                        {name: wandb.Audio(audio, sample_rate=sampling_rate)},
                        step=self.global_step,
                    )
                case loggers.TensorBoardLogger:
                    logger.experiment.add_audio(
                        name,
                        audio,
                        self.global_step,
                        sampling_rate,
                    )




