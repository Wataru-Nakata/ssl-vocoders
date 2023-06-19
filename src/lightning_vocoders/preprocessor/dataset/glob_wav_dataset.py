import torch
from torch.utils.data.dataset import Dataset
import torchaudio
from pathlib import Path
import random
import string

def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

class GlobWavDataset(Dataset):
    def __init__(self, roots, patterns, shuffled: bool = True,add_random_string=True) -> None:
        self.wav_files = []
        for root,pattern in zip(roots,patterns):
            self.root = Path(root)
            self.wav_files.extend(list(self.root.glob(pattern)))
        if shuffled:
            random.shuffle(self.wav_files)
        self.add_random_string = add_random_string

    def __len__(self):
        return len(self.wav_files)


    def __getitem__(self,idx):
        wav_path = self.wav_files[idx]
        if self.add_random_string:
            return wav_path.stem + generate_random_string(5), torchaudio.load(wav_path),str(wav_path)
        else:
            return wav_path.stem , torchaudio.load(wav_path),str(wav_path)
