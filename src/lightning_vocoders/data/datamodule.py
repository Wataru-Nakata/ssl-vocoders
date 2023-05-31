import webdataset as wds
import lightning
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import torch
import json


class VocoderDataModule(lightning.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        self.train_dataset = (
            wds.WebDataset(self.cfg.data.train_dataset_path)
            .shuffle(1000)
            .decode(wds.torch_audio)
        )
        self.val_dataset = (
            wds.WebDataset(self.cfg.data.val_dataset_path)
            .shuffle(1000)
            .decode(wds.torch_audio)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=20,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=20,
        )

    def collate_fn(self, batch):
        outputs = dict()
        outputs["mels"] = pad_sequence(
            [b["mel.pth"].T for b in batch], batch_first=True
        )

        outputs["mel_lens"] = torch.tensor([b["mel.pth"].size(1) for b in batch])
        outputs["max_mel_len"] = outputs["mel_lens"].max()

        outputs["ssl_feature"] = pad_sequence(
            [b[self.cfg.data.target_feature].T for b in batch], batch_first=True
        )
        return outputs
