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
python3 src/train.py preprocess=wavlm_large model=hifigan_ssl_large data=wavlm_large