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

Also, colab example in provided for your better understanding. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-Rj6eBGc-0owr8q1u7KR9ca0V20ws8n4?usp=sharing)
# Provieded checkpoints

|SSL model name | layer 3 | layer 6| layer 9 | layer 12 | layer 24 |
|---|---|---|---|---|---|
| wav2vec2-base | ☑️   | ☑️   | ☑️   | ☑️   | N/A |
| wav2vec2-large |  ❌  | ❌  | ☑️   | ❌  |  ☑️   |
| hubert-base |  ☑️   |   ❌  |  ❌  |  ☑️   |  ❌  |
| hubert-large |  ❌  | ☑️   | ❌  | ❌  |  ☑️   |
| wavlm-base |  ☑️   |   ❌  |  ❌  |  ☑️   |  ❌  |
| wavlm-large |  ❌  | ☑️   | ❌  | ❌  |  ☑️   |




 

# Acknoledgements
I'd like to express my sincere gratitude to to 
* jlk876's HiFiGAN [paper](https://arxiv.org/abs/2010.05646) and their [official implementation](https://github.com/jik876/hifi-gan)

