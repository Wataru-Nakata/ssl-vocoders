python3 src/preprocess.py preprocess=wav2vec2-base > wavlm_base.log &
python3 src/preprocess.py preprocess=wav2vec2-large > wavlm_large.log &
python3 src/preprocess.py preprocess=wavlm-base > wavlm_base.log &
python3 src/preprocess.py preprocess=wavlm-large > wavlm_large.log &
python3 src/preprocess.py preprocess=hubert-base > hubert_base.log &
python3 src/preprocess.py preprocess=hubert-large > hubert_large.log  