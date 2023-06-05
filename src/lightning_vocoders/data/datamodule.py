import webdataset as wds
import lightning
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import torch
from torch.utils.data import random_split
import json
import math
import random
import torchaudio
import transformers
import hydra

class VocoderDataModule(lightning.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        dataset = hydra.utils.instantiate(self.cfg.preprocess.preprocess_dataset)
        self.train_dataset,self.val_dataset = random_split(dataset,[0.9,0.1])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=lambda batch: self.collate_fn(
                batch, self.cfg.model.train_segment_size
            ),
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

    @torch.no_grad()
    def collate_fn(self, batch, segment_size: int = -1):

        outputs = dict()
        if segment_size != -1:
            cropped_speeches = []
            for sample in batch:
                file_name, (wav,sr) = sample
                audio_len = wav.size(1)
                if audio_len >= segment_size:
                    speech_start = random.randint(
                        0, audio_len - segment_size - 1
                    )
                    cropped_speeches.append(
                        wav.squeeze()[
                            speech_start : speech_start+segment_size
                        ]
                    )
                else:
                    cropped_speeches.append(wav.squeeze())
            outputs["resampled_speech.pth"] = pad_sequence(
                cropped_speeches, batch_first=True
            )
        else:
            outputs["resampled_speech.pth"] = pad_sequence(
                [b[1][0].squeeze() for b in batch], batch_first=True
            )
        
        outputs["wav_lens"] = torch.tensor(
            [b[1][0].size(1) for b in batch]
        )

        outputs["filenames"] = [b[0] for b in batch]
        return outputs

    def calc_spectrogram(self, waveform: torch.Tensor):
        magspec = self.spec_module(waveform)
        melspec = self.mel_scale(magspec)
        logmelspec = torch.log(torch.clamp_min(melspec, 1.0e-5) * 1.0).to(torch.float32)
        energy = torch.norm(magspec, dim=0)
        return logmelspec, energy
