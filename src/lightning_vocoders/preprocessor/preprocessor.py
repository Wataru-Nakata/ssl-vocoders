import torch
import hydra
import torchaudio
import pathlib
from omegaconf import DictConfig
import numpy as np
import webdataset
import tqdm


class Preprocessor:
    """
    Preprocess dataset
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: hydra config
        """
        self.cfg = cfg
        self.dataset = hydra.utils.instantiate(cfg.preprocess.preprocess_dataset)
        self.spec_module = torchaudio.transforms.Spectrogram(**cfg.preprocess.stft)
        self.mel_scale = torchaudio.transforms.MelScale(**cfg.preprocess.mel)
        self.sampling_rate = self.cfg.sample_rate
        self.ssl_models = hydra.utils.instantiate(cfg.preprocess.ssl_models)

    @torch.no_grad()
    def process_utterance(
        self,
        basename: str,
        audio_file_path: pathlib.Path,
    ):
        orig_waveform, sample_rate = torchaudio.load(audio_file_path)

        waveform = torchaudio.functional.resample(
            orig_waveform, sample_rate, new_freq=self.sampling_rate
        )[
            0
        ]  # remove channel dimension only support mono

        mel_spec, energy = self.calc_spectrogram(waveform)
        with open(audio_file_path, mode="rb") as f:
            wav_bytes = f.read()

        sample = {
            "__key__": basename,
            "speech.wav": wav_bytes,
            "resampled_speech.pth": webdataset.torch_dumps(waveform),
            "mel.pth": webdataset.torch_dumps(mel_spec.T),
        }
        for ssl_model, processor, feature_cfg in self.ssl_models:
            wav_tensor = torchaudio.functional.resample(
                waveform=orig_waveform, orig_freq=sample_rate, new_freq=feature_cfg.sr
            )
            inputs = processor(
                wav_tensor.squeeze(), return_tensors="pt", sampling_rate=feature_cfg.sr
            )
            inputs.to("cuda")
            ssl_model.to("cuda")
            output = ssl_model(**inputs, output_hidden_states=True)
            sample[feature_cfg.key] = webdataset.torch_dumps(
                output.hidden_states[feature_cfg.layer][0].cpu()
            )

        return sample

    def build_from_path(self):
        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)
        for idx, (basename, wav_file_path) in tqdm.tqdm(enumerate(self.dataset)):
            sample = self.process_utterance(
                basename,
                wav_file_path,
            )
            if idx >= self.cfg.preprocess.val_size:
                train_sink.write(sample)
            else:
                val_sink.write(sample)

        train_sink.close()
        val_sink.close()

    def calc_spectrogram(self, waveform: torch.Tensor):
        magspec = self.spec_module(waveform)
        melspec = self.mel_scale(magspec)
        logmelspec = torch.log(torch.clamp_min(melspec, 1.0e-5) * 1.0).to(torch.float32)
        energy = torch.norm(magspec, dim=0)
        return logmelspec, energy.numpy()
