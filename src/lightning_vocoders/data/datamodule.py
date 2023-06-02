import webdataset as wds
import lightning
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import torch
import json
import math
import random


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
        self.val_dataset = wds.WebDataset(self.cfg.data.val_dataset_path).decode(
            wds.torch_audio
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=lambda batch: self.collate_fn(
                batch, self.cfg.model.train_segment_size
            ),
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )
    @torch.no_grad()
    def collate_fn(self, batch, segment_size: int = -1):
        outputs = dict()
        if segment_size != -1:
            input_feature_hop_size = (
                self.cfg.sample_rate / self.cfg.data.target_feature.samples_per_sec
            )
            mel_hop_size = self.cfg.preprocess.stft.hop_length
            target_feature = self.cfg.data.target_feature.key
            target_layer = self.cfg.data.target_feature.layer
            input_feature_frames_per_seg = math.ceil(
                segment_size / input_feature_hop_size
            )
            mel_frames_per_seg = math.ceil(segment_size / mel_hop_size)
            cropped_speeches = []
            cropped_features = []
            cropped_mels = []
            for sample in batch:
                audio_len = sample["resampled_speech.pth"].size(0)
                data_len = sample[target_feature].size(1)
                if sample["resampled_speech.pth"].size(0) >= segment_size:
                    input_feature_start = random.randint(
                        0, data_len - input_feature_frames_per_seg - 1
                    )
                    mel_start = int(
                        input_feature_start * input_feature_hop_size / mel_hop_size
                    )
                    cropped_features.append(
                        sample[target_feature][target_layer][
                            input_feature_start : input_feature_start
                            + input_feature_frames_per_seg,:
                            
                        ]
                    )
                    cropped_speeches.append(
                        sample["resampled_speech.pth"][
                            int(input_feature_start * input_feature_hop_size) : int(
                                (input_feature_start + input_feature_frames_per_seg)
                                * input_feature_hop_size
                            )
                        ]
                    )
                    cropped_mels.append(
                        sample["mel.pth"][0][
                            mel_start : mel_start + mel_frames_per_seg,:
                        ]
                    )
                else:
                    cropped_features.append(sample[target_feature][target_layer])
                    cropped_speeches.append(sample["resampled_speech.pth"])
                    cropped_mels.append(sample["mel.pth"][0])
            outputs["resampled_speech.pth"] = pad_sequence(
                cropped_speeches, batch_first=True
            )
            outputs["ssl_feature"] = pad_sequence(cropped_features, batch_first=True)
            outputs["mels"] = pad_sequence(
                cropped_mels, batch_first=True, padding_value=1e-9
            )
            outputs["mel_lens"] = torch.tensor(
                [cropped_mel.size(1) for cropped_mel in cropped_mels]
            )
            outputs["max_mel_len"] = outputs["mel_lens"].max()
        else:
            outputs["resampled_speech.pth"] = pad_sequence(
                [b["resampled_speech.pth"] for b in batch], batch_first=True
            )
            outputs["ssl_feature"] = pad_sequence(
                [
                    b[self.cfg.data.target_feature.key][
                        self.cfg.data.target_feature.layer
                    ]
                    for b in batch
                ],
                batch_first=True,
            )
            outputs["mels"] = pad_sequence(
                [b["mel.pth"][0] for b in batch], batch_first=True, padding_value=1e-9
            )
            outputs["mel_lens"] = torch.tensor([b["mel.pth"].size(0) for b in batch])
            outputs["max_mel_len"] = outputs["mel_lens"].max()
        outputs["wav_lens"] = torch.tensor(
            [b["resampled_speech.pth"].size(0) for b in batch]
        )

        outputs["speech.wav"] = pad_sequence(
            [b["speech.wav"][0][0] for b in batch], batch_first=True
        )
        outputs["filenames"] = [b["__key__"] for b in batch]
        return outputs
