import hydra
import torch
from lightning.pytorch import LightningModule
MODEL_URLS = {
    "hubert-base-l3": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/hubert-base-l3/model.ckpt",
    "hubert-base": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/hubert-base/model.ckpt",
    "hubert-large-l6": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/hubert-large-l6/model.ckpt",
    "hubert-large": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/hubert-large/model.ckpt",
    "mel": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/mel/model.ckpt",
    "wav2vec2-base": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2-base/model.ckpt",
    "wav2vec2-large": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2-large/model.ckpt",
    "wav2vec2_l3": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2_l3/model.ckpt",
    "wav2vec2_l6": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2_l6/model.ckpt",
    "wav2vec2_l9": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wav2vec2_l9/model.ckpt",
    "wavlm-base-l3": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wavlm-base-l3/model.ckpt",
    "wavlm-large-l6": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wavlm-large-l6/model.ckpt",
    "wavlm-large": "https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wavlm-large/model.ckpt",
}
