import argparse
import lightning.pytorch as pl
import torch
import io
from omegaconf import DictConfig
import hydra
from pathlib import Path
from lightning_vocoders.preprocessor.preprocessor import Preprocessor
from lightning_vocoders.preprocessor.dataset.glob_wav_dataset import GlobWavDataset
from torch.utils.data.dataloader import DataLoader

def synthesize(cfg:DictConfig,ckpt_path:Path,wav_path:Path):
    lightning_module:pl.LightningModule = hydra.utils.instantiate(cfg.model.lightning_module,cfg)
    lightning_module = lightning_module.load_from_checkpoint(ckpt_path)

    dataset  = GlobWavDataset([wav_path],["**/*.wav"],shuffled=False,add_random_string=False)
    preprocessor = Preprocessor(lightning_module.cfg)

    @torch.no_grad()
    def test_collate_fn(sample):
        assert len(sample) == 1 # only expect batch size of 1
        wav_name, (wav_data,sr), wav_path = sample[0]
        preprocessed_sample = preprocessor.process_utterance(wav_name,wav_data,sr,wav_path)
        for k,v in preprocessed_sample.items():
            if k.endswith(".pth"):
                preprocessed_sample[k] = torch.load(io.BytesIO(v))
        batch = {
            "resampled_speech.pth": None,
            "input_feature": preprocessed_sample[cfg.data.target_feature.key].unsqueeze(0),
            "filenames": [preprocessed_sample["__key__"]],
            "wav_lens": None
        }
        return batch
    test_dataloader = DataLoader(dataset,collate_fn=test_collate_fn)
    trainer = pl.Trainer()
    trainer.test(lightning_module,test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path",required=True)
    parser.add_argument("--wav", required=True,type=str)
    args = parser.parse_args()
    ckpt = torch.load(args.ckpt_path)

    cfg = ckpt['hyper_parameters']['cfg']


    synthesize(cfg,args.ckpt_path,args.wav)
