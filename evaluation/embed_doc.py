from content_memory import OpenaiEmbedder, ContrieverEmbedder, MemoryDoc, MemoryBuffer
from utils import load_embed_model_by_name
from pathlib import Path
import json
from tqdm import tqdm
import os
import re
import fire

def generate_doc_embedding(embedding_save_path:str, file_path:str, embed_model_name:str) -> None:
    embed_model = load_embed_model_by_name(embed_model_name)
    memory_buffer = MemoryBuffer(file_path=embedding_save_path, embed_model=embed_model)
    with open(file_path, 'r') as f:
        data = json.load(f)
    for d in tqdm(data):
        memory_buffer.add_memory(d, time=None)
        memory_buffer.save_memory_to_binary()

if __name__ == "__main__":
    fire.Fire(generate_doc_embedding)