import torch
from torch.utils.data.dataset import Dataset
import torchaudio
from pathlib import Path
import random


class GlobWavDataset(Dataset):
    def __init__(self, root, pattern, shuffled: bool = True) -> None:
        self.root = Path(root)
        self.wav_files = list(self.root.glob(pattern))
        if shuffled:
            random.shuffle(self.wav_files)

    def __len__(self):
        return len(self.wav_files)


    def __getitem__(self,idx):
        wav_path = self.wav_files[idx]
        return wav_path.stem, torchaudio.load(wav_path)
