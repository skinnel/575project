#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh

conda activate news_emo

python probe.py --dataset silver --load
 