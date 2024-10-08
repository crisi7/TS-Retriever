## Time-sensitive Retrieval-Augmented Generation for Question Answering
This repository contains sft models, benchmark dataset and code for evaluation for our paper.

## Dataset
``evaluation/data/nobel_prize``: this folder contains the test benchmark dataset, including queries and corpus.
``train/dataset/sft/``: this folder contains the train dataset to finetune the contriever.
## SFT models
The following models are available:
* Tscontriever: The finetuned model using positive and negative sample pairs with temporal constraints.
* Tscontriever_query_only: the query-side finetuned model.
* Router: A simple classifier is used to route the query.

The model weights can be downloaded from [this Baidu Netdisk link](https://pan.baidu.com/s/1Rwyo7N9lyo6H0OON97c5mg?pwd=9f5m).
## Evaluation
We pre-computed and stored the embeddings of the query and the documents to be retrieved. You can reproduce the results on our benchmark using the following command.

The embeddings can be downloaded from [this Baidu Netdisk link](https://pan.baidu.com/s/1Rwyo7N9lyo6H0OON97c5mg?pwd=9f5m).

* For Tscontriever result:
```bash
cd evaluation
embed_model_query="Tscontriever"
embed_model_doc="Tscontriever"
query_embed_save_dir="./temp_embed_files/query"
doc_embed_save_dir="./temp_embed_files/docs"
python experiment.py "${embed_model_query}" "${embed_model_doc}" "${query_embed_save_dir}" "${doc_embed_save_dir}/${embed_model_doc}_embed_doc"
```
* For Tscontriever_query_only result:
```bash
cd evaluation
embed_model_query="Tscontriever_query_only"
embed_model_doc="Tscontriever"
query_embed_save_dir="./temp_embed_files/query"
doc_embed_save_dir="./temp_embed_files/docs"
python experiment.py "${embed_model_query}" "${embed_model_doc}" "${query_embed_save_dir}" "${doc_embed_save_dir}/${embed_model_doc}_embed_doc"
```
* For Tscontriever_query_only_with_router result:
```bash
cd evaluation
embed_model_query="Tscontriever_with_router"
embed_model_doc="Tscontriever"
query_embed_save_dir="./temp_embed_files/query"
doc_embed_save_dir="./temp_embed_files/docs"
python experiment.py "${embed_model_query}" "${embed_model_doc}" "${query_embed_save_dir}" "${doc_embed_save_dir}/${embed_model_doc}_embed_doc"
```

Alternatively, You can also download the model weights and encode the query and documents to be retrieved to reproduce the results,Follow these steps:
1. Download the model weights and place them in the `evaluation/models` folder.
2. Navigate to the `evaluation` directory.
3. Run the command: `bash ./eval.sh`.

## Training
The training code is based on the contriever repository with slightly modified. To train the model, you can use the following command.
* For Tscontriever:
```bash
python ./train/contriever/contriever/finetuning.py \
    --model_path <your contriever model path> \
    --eval_data ./train/dataset/sft/contriever_finetune_eval_v3.jsonl \
    --train_data ./train/dataset/sft/contriever_finetune_train_v3.jsonl \
    --save_freq 5000 \
    --eval_freq 100 \
    --random_init false \
    --total_steps 1500 \
    --negative_ctxs 1
```
* For Tscontriever_query_only:
```bash
python ./train/contriever/finetuning_frozen.py \
    --model_path <your contriever model path> \
    --eval_data ./train/dataset/sft/contriever_finetune_eval_v3.jsonl \
    --train_data ./train/dataset/sft/contriever_finetune_train_v3.jsonl \
    --save_freq 5000 \
    --eval_freq 100 \
    --random_init false \
    --total_steps 1500 \
    --negative_ctxs 1
```