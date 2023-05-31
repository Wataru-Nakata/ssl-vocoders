import torch
import torchaudio
from pathlib import Path
import random


class GlobWavDataset:
    def __init__(self, root, pattern, shuffled: bool = True) -> None:
        self.root = Path(root)
        self.wav_files = list(self.root.glob(pattern))
        if shuffled:
            random.shuffle(self.wav_files)
        self.current_wav_index = 0

    def __len__(self):
        return len(self.wav_files)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            wav_path = self.wav_files[self.current_wav_index]
            self.current_wav_index += 1
            return wav_path.stem, wav_path
        except IndexError:
            raise StopIteration
