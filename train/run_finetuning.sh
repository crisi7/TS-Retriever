#!/bin/bash

python ~/TS-Retriever/train/contriever/finetuning.py \
    --model_path facebook/contriever \
    --eval_data ~/TS-Retriever/train/dataset/sft/contriever_finetune_eval_v3.jsonl \
    --train_data ~/TS-Retriever/train/dataset/sft/contriever_finetune_train_v3.jsonl \
    --save_freq 5000 \
    --eval_freq 100 \
    --random_init false \
    --total_steps 1500 \
    --negative_ctxs 1
