import hydra
import torch
from lightning.pytorch import LightningModule
MODEL_URLS = {
    "hubert-base-l3": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/hubert-base-l3/hubert-base-l3.ckpt",
    "hubert-base": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/hubert-base/hubert-base.ckpt",
    "hubert-large-l6": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/hubert-large-l6/hubert-large-l6.ckpt",
    "hubert-large": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/hubert-large/hubert-large.ckpt",
    "mel": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/mel/mel.ckpt",
    "wav2vec2-base": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2-base/wav2vec2-base.ckpt",
    "wav2vec2-large": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2-large/wav2vec2-large.ckpt",
    "wav2vec2-large-l6": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2-large-l6/wav2vec2-large-l6.ckpt",
    "wav2vec2_l3": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2_l3/wav2vec2_l3.ckpt",
    "wav2vec2_l6": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2_l6/wav2vec2_l6.ckpt",
    "wav2vec2_l9": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2_l9/wav2vec2_l9.ckpt",
    "wavlm-base-l3": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wavlm-base-l3/wavlm-base-l3.ckpt",
    "wavlm-large-l6": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wavlm-large-l6/wavlm-large-l6.ckpt",
    "wavlm-large": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wavlm-large/wavlm-large.ckpt",
}
