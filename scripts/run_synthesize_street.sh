python3 src/synthesize.py --ckpt_path notebooks/checkpoints/wav2vec2-base/model.ckpt  --wav /mnt/hdd/datasets/NOIZEUS/15dB --output_path street/wav2vec2-base/
python3 src/synthesize.py --ckpt_path notebooks/checkpoints/wav2vec2-large/model.ckpt --wav /mnt/hdd/datasets/NOIZEUS/15dB --output_path street/wav2vec2-large/
python3 src/synthesize.py --ckpt_path notebooks/checkpoints/hubert-base/model.ckpt    --wav /mnt/hdd/datasets/NOIZEUS/15dB --output_path street/hubert-base/
python3 src/synthesize.py --ckpt_path notebooks/checkpoints/hubert-large/model.ckpt   --wav /mnt/hdd/datasets/NOIZEUS/15dB --output_path street/hubert-large/
python3 src/synthesize.py --ckpt_path notebooks/checkpoints/wavlm-base/model.ckpt     --wav /mnt/hdd/datasets/NOIZEUS/15dB --output_path street/wavlm-base/
python3 src/synthesize.py --ckpt_path notebooks/checkpoints/wavlm-large/model.ckpt    --wav /mnt/hdd/datasets/NOIZEUS/15dB --output_path street/wavlm-large/
python3 src/synthesize.py --ckpt_path notebooks/checkpoints/mel/model.ckpt            --wav /mnt/hdd/datasets/NOIZEUS/15dB --output_path street/mel/
