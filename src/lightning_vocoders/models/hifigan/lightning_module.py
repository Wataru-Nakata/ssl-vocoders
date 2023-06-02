import itertools
from typing import Any, Optional
from lightning.pytorch import LightningModule, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import numpy as np
from .hifigan import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    generator_loss,
    feature_loss,
)
import torch
import hydra
import torchaudio


class HiFiGANLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.generator = Generator(cfg.model.generator)
        self.multi_period_discriminator = MultiPeriodDiscriminator()
        self.multi_scale_discriminator = MultiScaleDiscriminator()
        self.automatic_optimization = False
        self.spec_module = torchaudio.transforms.Spectrogram(**cfg.preprocess.stft)
        self.mel_scale = torchaudio.transforms.MelScale(**cfg.preprocess.mel)
        self.cfg = cfg
        self.save_hyperparameters()

    def configure_optimizers(self) -> Any:
        opt_g = hydra.utils.instantiate(
            self.cfg.model.optim.opt_g, params=self.generator.parameters()
        )
        opt_d = hydra.utils.instantiate(
            self.cfg.model.optim.opt_d,
            params=itertools.chain(
                self.multi_scale_discriminator.parameters(),
                self.multi_period_discriminator.parameters(),
            ),
        )
        scheduler_g = hydra.utils.instantiate(
            self.cfg.model.optim.scheduler_g, optimizer=opt_g
        )
        scheduler_d = hydra.utils.instantiate(
            self.cfg.model.optim.scheduler_d, optimizer=opt_d
        )

        return [opt_g, opt_d], [
            {"name": "scheduler_g", "scheduler": scheduler_g},
            {"name": "scheduler_d", "scheduler": scheduler_d},
        ]

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        generator_input, wav, mel, _ = (
            batch["ssl_feature"],
            batch["resampled_speech.pth"],
            batch["mels"],
            batch["filenames"],
        )
        wav = wav.unsqueeze(1)
        wav_generator_out = self.generator(generator_input)

        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        if self.global_step >= self.cfg.model.adversarial_start_step:
            opt_d.zero_grad()

            # mpd
            mpd_out_real, mpd_out_fake, _, _ = self.multi_period_discriminator(
                wav, wav_generator_out.detach()
            )
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                mpd_out_real, mpd_out_fake
            )

            # msd
            msd_out_real, msd_out_fake, _, _ = self.multi_scale_discriminator(
                wav, wav_generator_out.detach()
            )
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                msd_out_real, msd_out_fake
            )

            loss_disc_all = loss_disc_s + loss_disc_f
            self.manual_backward(loss_disc_all)
            opt_d.step()
            sch_d.step()
            self.log("train/discriminator/loss_disc_f", loss_disc_f)
            self.log("train/discriminator/loss_disc_s", loss_disc_s)
        else:
            loss_disc_f = loss_disc_s = 0.0

        # generator
        opt_g.zero_grad()
        predicted_mel, _ = self.calc_spectrogram(wav_generator_out)
        loss_recons = self.reconstruction_loss(mel, predicted_mel)
        loss_g = loss_recons * self.cfg.model.loss.recons_coef
        if self.global_step >= self.cfg.model.adversarial_start_step:
            mpd_out_real, mpd_out_fake, _, _ = self.multi_period_discriminator(
                wav, wav_generator_out
            )
            loss_fm_mpd = feature_loss(mpd_out_real, mpd_out_fake)

            # msd
            msd_out_real, msd_out_fake, _, _ = self.multi_scale_discriminator(
                wav, wav_generator_out
            )
            loss_fm_msd = feature_loss(msd_out_real, msd_out_fake)

            loss_g_mpd, losses_gen_f = generator_loss(mpd_out_fake)
            loss_g_msd, losses_gen_s = generator_loss(msd_out_fake)
            loss_g += loss_fm_mpd * self.cfg.model.loss.fm_mpd_coef
            loss_g += loss_fm_msd * self.cfg.model.loss.fm_msd_coef
            loss_g += loss_g_mpd * self.cfg.model.loss.g_mpd_coef
            loss_g += loss_g_msd * self.cfg.model.loss.g_msd_coef
            self.log("train/generator/loss_fm_mpd", loss_fm_mpd)
            self.log("train/generator/loss_fm_msd", loss_fm_msd)
            self.log("train/generator/loss_g_mpd", loss_g_mpd)
            self.log("train/generator/loss_g_msd", loss_g_msd)
        self.manual_backward(loss_g)
        self.log("train/loss_reconstruction", loss_recons)
        self.log("train/generator/loss", loss_g)
        opt_g.step()
        sch_g.step()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        generator_input, wav, mel, filename = (
            batch["ssl_feature"],
            batch["resampled_speech.pth"],
            batch["mels"],
            batch["filenames"],
        )
        wav_generator_out = self.generator(generator_input)
        predicted_mel, _ = self.calc_spectrogram(wav_generator_out)
        loss_recons = self.reconstruction_loss(mel, predicted_mel)
        if (
            batch_idx < self.cfg.model.logging_wav_samples
            and self.global_rank == 0
            and self.local_rank == 0
        ):
            self.log_audio(
                wav_generator_out[0].squeeze().cpu().numpy().astype(np.float32),
                name=f"generated/{filename[0]}",
                sampling_rate=self.cfg.sample_rate,
            )
            self.log_audio(
                wav[0].squeeze().cpu().numpy().astype(np.float32),
                name=f"natural/{filename[0]}",
                sampling_rate=self.cfg.sample_rate,
            )

        self.log("val/reconstruction", loss_recons)

    def reconstruction_loss(self, mel_gt, mel_predicted):
        return torch.nn.L1Loss()(
            mel_gt, mel_predicted.squeeze(1).transpose(1, 2)[:, : mel_gt.size(1), :]
        )

    def calc_spectrogram(self, waveform: torch.Tensor):
        magspec = self.spec_module(waveform)
        melspec = self.mel_scale(magspec)
        logmelspec = torch.log(torch.clamp_min(melspec, 1.0e-5) * 1.0).to(torch.float32)
        energy = torch.norm(magspec, dim=0)
        return logmelspec, energy

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
