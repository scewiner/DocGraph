#!/usr/bin/bash

# set -x
data_dir=~/data/IWSLT/zh-en/SAN-zhen-iwslt
save_dir=~/models/IWSLT-zhen-eval/ 
log_dir=~/log/IWSLT-zhen-eval/
SPM=~/models/IWSLT/zh-en/aver/checkpoint_sent.pt

user_dir=./
output=${output}/inference-bpe
pip install dgl
# train forward graph which based on unified encoder, Ma et al acl2020
fairseq-train $data_dir \
        --user-dir $user_dir \
        -s zh -t en \
        --task DocGraphTask \
        --arch DGFoward \
        --save-dir ${save_dir} \
        --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt --lr 7e-05 \
        --warmup-updates 16000  \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 --update-freq 1 \
        --encoder-normalize-before  --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.00001 \
        --max-update 30000 \
        --pretrained-checkpoint $SPM \
        --shared-layers \
        --typeatt \
        --share-ctx-layer \
        --log-format simple \
        --ddp-backend=no_c10d | tee -a "${log_dir}/eval-dgfw.log"

