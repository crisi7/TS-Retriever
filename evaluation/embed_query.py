from content_memory import OpenaiEmbedder, MemoryDoc, ContrieverEmbedder
from pathlib import Path
from utils import load_embed_model_by_name, read_json, save_json
import json
from tqdm import tqdm
import os
import re
import fire

def embed_query(save_dir_path, query_filepath, embed_model_name):
    embedder = load_embed_model_by_name(embed_model_name)
    filepath = Path(query_filepath)
    filename = filepath.stem
    savepath = Path(save_dir_path) / f"{embed_model_name}_{filename}_embbed.json"
    data = read_json(filepath)
    for d in tqdm(data, desc="Embedding queries"):
        if 'embedding' not in d:
            d['embedding'] = embedder.embedding(content=d['query'], return_type='json')
    save_json(data, savepath)

def batch_embed_query(save_dir_path, query_filepath, embed_model_name, batch_size=16):
    embedder = load_embed_model_by_name(embed_model_name)
    filepath = Path(query_filepath)
    filename = filepath.stem
    savepath = Path(save_dir_path) / f"{embed_model_name}_{filename}_embbed.json"
    data = read_json(filepath)
    queries_batch = []
    start_index = 0
    for d in tqdm(data, desc="Embedding queries"):
        queries_batch.append(d['query'])
        if len(queries_batch) == batch_size:
            embeddings = embedder.batch_embedding(queries_batch, return_type='json')
            for i, emb in enumerate(embeddings):
                data[start_index + i]['embedding'] = emb
            start_index += len(queries_batch)
            queries_batch.clear()
    if queries_batch:
        embeddings = embedder.batch_embedding(queries_batch, return_type='json')
        start_index = len(data) - len(queries_batch)
        for i, emb in enumerate(embeddings):
            data[start_index + i]['embedding'] = emb
    save_json(data, savepath)

if __name__ == "__main__":
    fire.Fire(batch_embed_query)