source /etc/profile.d/modules.sh
module load python/3.10/3.10.10
module load cuda/12.1/12.1.1
module load cudnn/8.9/8.9.2
module load nccl/2.18/2.18.1-1
source venv/bin/activate
python3 src/train.py preprocess=wav2vec2-base model=hifigan_ssl data=wav2vec2_base data.train_dataset_path=/scratch/acc12576tt/wav2vec2_base/wav2vec2_base-train-{000000..000119}.tar.gz data.val_dataset_path=/scratch/acc12576tt/wav2vec2_base/wav2vec2_base-val-{000000..000001}.tar.gz