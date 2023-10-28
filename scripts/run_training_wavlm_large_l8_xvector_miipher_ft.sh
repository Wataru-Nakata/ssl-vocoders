#! /bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$-cwd
source /etc/profile.d/modules.sh
module load python/3.10/3.10.10
module load cuda/12.1/12.1.1
module load cudnn/8.9/8.9.2
module load nccl/2.18/2.18.1-1
source venv/bin/activate
python3 src/train.py data=wavlm_large_l8_xvector \
	model=hifigan_ssl_large_xvector \
	preprocess=wavlm_large_l8 \
	data.target_feature.key="miipher_cleaned_feature.pth" \
	data.train_batch_size=16 \
	'+train.trainer.val_check_interval=10000' \
	train.trainer.check_val_every_n_epoch=null \
	model.optim.opt_g.lr=0.00002 \
	model.optim.opt_d.lr=0.00002 \
	model.adversarial_start_step=0 \
	'data.train_dataset_path="/mnt/hdd/finetune_train.tar.gz"' \
	'data.val_dataset_path="/mnt/hdd/finetune_val.tar.gz"' \
	'train.ckpt_path="https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wavlm-large-l8-xvector/wavlm-large-l8-xvector.ckpt"'
