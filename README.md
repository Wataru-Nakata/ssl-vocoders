# ssl-vocoders
This repository contains training script for creating vocoders from speech Self-supervised learning models features.

# Installation
```bash
git clone https://github.com/Wataru-Nakata/ssl-vocoders
cd ssl-vocoders
pip install -e .
```

# Pretrained models
Pretrained models are distributed on [huggingface](https://huggingface.co/Wataru/ssl-vocoder/tree/main)

# How to use the pretrained models
To load hifigan model trained on wavlm-large final hidden layer feature, try running the code below.
```python
import lightning_vocoders
from lightning_vocoders.models.hifigan.lightning_module import HiFiGANLightningModule
model = HiFiGANLightningModule.load_from_checkpoint(lightning_vocoders.MODEL_URLS['wavlm-large'],map_location='cpu')
```
# Provieded checkpoints

|SSL model name | layer 3 | layer 6| layer 9 | layer 12 | layer 24 |
|---|---|---|---|---|---|
| wav2vec2-base | ☑️   | ☑️   | ☑️   | ☑️   | N/A |
| wav2vec2-large |  ❌  | ❌  | ☑️   | ❌  |  ☑️   |
| hubert-base |  ☑️   |   ❌  |  ❌  |  ☑️   |  ❌  |
| hubert-large |  ❌  | ☑️   | ❌  | ❌  |  ☑️   |
| wavlm-base |  ☑️   |   ❌  |  ❌  |  ☑️   |  ❌  |
| wavlm-large |  ❌  | ☑️   | ❌  | ❌  |  ☑️   |




 

