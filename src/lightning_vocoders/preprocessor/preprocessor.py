import torch
import hydra
import torchaudio
import pathlib
from omegaconf import DictConfig
import numpy as np
import webdataset
import tqdm
from torch.utils.data import DataLoader


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
        if hasattr(cfg.data, "xvector"):
            self.use_xvector = cfg.data.xvector.use_xvector
        else:
            self.use_xvector = False
        if self.use_xvector:
            self.xvector_model = hydra.utils.instantiate(self.cfg.data.xvector.model)
            self.xvector_model.eval()
            self.xvector_sr = self.cfg.data.xvector.sr
            self.xvector_extract_secs = self.cfg.data.xvector.extract_secs
            self.xvector_embedding_size = self.cfg.data.xvector.embedding_size

    @torch.no_grad()
    def process_utterance(
        self,
        basename: str,
        orig_waveform: torch.Tensor,
        sample_rate: int,
        audio_file_path
    ):

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
            if processor is not None:
                inputs = processor(
                    wav_tensor.squeeze(), return_tensors="pt", sampling_rate=feature_cfg.sr
                )
                inputs.to("cuda")
                ssl_model.to("cuda")
                output = ssl_model(**inputs, output_hidden_states=True)
                sample[feature_cfg.key] = webdataset.torch_dumps(
                    output.hidden_states[feature_cfg.layer][0].cpu()
                )
            else:
                ssl_model.to("cuda")
                wav_tensor = wav_tensor.unsqueeze(1).to('cuda')
                output = ssl_model({"x": wav_tensor},sample_rate=feature_cfg.sr)
                sample[feature_cfg.key] = webdataset.torch_dumps(
                    output.hidden_states[feature_cfg.layer][0].cpu()
                )
        if self.use_xvector:
            resampled_for_xvector = torchaudio.functional.resample(
                    orig_waveform, sample_rate, self.xvector_sr
                ).squeeze()[: int(self.xvector_sr * self.xvector_extract_secs)]
            embeddings = self.xvector_model.encode_batch(resampled_for_xvector.unsqueeze(0))
            sample["xvector.pth"] = embeddings.view(self.xvector_embedding_size)


        return sample

    def build_from_path(self):
        pathlib.Path("/".join(self.cfg.preprocess.train_tar_sink.pattern.split("/")[:-1])).mkdir(exist_ok=True)
        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)
        dataloader = DataLoader(self.dataset,batch_size=1)
        for idx, (basename, (wav,sr),wav_path) in enumerate(tqdm.tqdm(dataloader)):
            sample = self.process_utterance(
                basename[0],
                wav[0],
                sr[0],
                wav_path[0]
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
