#!/usr/bin/env bash

EXP_DIR=exps/voc_1
BASE_TRAIN_DIR=${EXP_DIR}/base_train
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

python -u main.py \
    --dataset_file voc_base1 \
    --backbone resnet101 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 4 \
    --category_codes_cls_loss \
    --epoch 25 \
    --lr_drop_milestones 20 \
    --save_every_epoch 5 \
    --eval_every_epoch 5 \
    --output_dir ${BASE_TRAIN_DIR} \
# 2>&1 | tee ${BASE_TRAIN_DIR}/log.txts