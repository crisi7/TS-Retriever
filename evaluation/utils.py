from content_memory import MemoryBuffer, OpenaiEmbedder, ContrieverEmbedder, JinaaiEmbedder,TasbEmbedder, LoraContrieverEmbedder, RouterDualEncoderRetriever
import json

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data
    
def save_json(data, filepath):
    with open(filepath,'w') as f:
        json.dump(data, f, indent = 4, ensure_ascii=False)

def load_experiment_doc(save_dir_path):
    memory_buffer = MemoryBuffer(file_path=save_dir_path)
    memory_buffer.load_memory_from_binary()
    return memory_buffer

def load_query_file(query_file):
    return read_json(query_file)


def get_metric_at_k(k, true_items, predicted_items):
    if k <= 0:
        raise ValueError("k must be a positive integer within the range of predicted items")
    if k > len(predicted_items):
        predicted_top_k = predicted_items[:]
    else:
        predicted_top_k = predicted_items[:k]
    hits = sum(1 for item in predicted_top_k if item in true_items)
    precision_at_k = min(hits / k, 1)
    recall_at_k = hits / len(true_items) if true_items else 0
    return precision_at_k, recall_at_k

def load_embed_model_by_name(embed_model_name):
    embed_model_name_with_kwargs = {
        "contriever": {"model_name_or_path":"./models/contriever"},
        "Tscontriever_with_differ_dataset": {"model_name_or_path":"./models/differ_sizes/contriever_256"},
        "Tscontriever": {"model_name_or_path":"./models/Tscontriever"},
        "Tscontriever_query_only": {"model_name_or_path": "./models/Tscontriever_query_only"},
        "openai": {"api_base": None, "api_key": None},
        "Tscontriever_query_only_lora": {"model_name_or_path":"./models/Tscontriever", "lora_model_path": "./models/Tscontriever_query_only_lora"},
        "jinaai": {"model_name_or_path": "./models/jinaai/jina-embeddings-v2-base-en"},
        "tasb": {"model_name_or_path": "./models/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"},
        "Tscontriever_with_router": {"router_state_dict_path": "./models/router/model_state_dict.pth"}
    }
    if embed_model_name == "contriever":
        embed_model = ContrieverEmbedder(**embed_model_name_with_kwargs[embed_model_name])
    elif embed_model_name == "Tscontriever_with_differ_dataset":
        embed_model = ContrieverEmbedder(**embed_model_name_with_kwargs[embed_model_name])
    elif embed_model_name == "Tscontriever":
        embed_model = ContrieverEmbedder(**embed_model_name_with_kwargs[embed_model_name])
    elif embed_model_name == "Tscontriever_query_only":
        embed_model = ContrieverEmbedder(**embed_model_name_with_kwargs[embed_model_name])
    elif embed_model_name.lower() == "openai":
        embed_model = OpenaiEmbedder(**embed_model_name_with_kwargs[embed_model_name])
    elif embed_model_name.lower() == "jinaai":
        embed_model = JinaaiEmbedder(**embed_model_name_with_kwargs[embed_model_name])
    elif embed_model_name.lower() == "tasb":
        embed_model = TasbEmbedder(**embed_model_name_with_kwargs[embed_model_name])
    elif embed_model_name == 'Tscontriever_query_only_lora':
        embed_model = LoraContrieverEmbedder(**embed_model_name_with_kwargs[embed_model_name])
    elif embed_model_name == 'Tscontriever_with_router':
        doc_encoder = ContrieverEmbedder(**embed_model_name_with_kwargs['contriever'])
        ts_encoder = ContrieverEmbedder(**embed_model_name_with_kwargs['Tscontriever_query_only'])
        embed_model = RouterDualEncoderRetriever(query_encoder=ts_encoder,
                                                 passage_encoder=doc_encoder,
                                                 router_state_dict_path=embed_model_name_with_kwargs[embed_model_name]['router_state_dict_path'])
    else:
        raise ValueError(f"Not support this embed model: {embed_model_name}, you can try modifying the source code")
    return embed_model

if __name__ == '__main__':
    e = load_embed_model_by_name('Tscontriever_with_differ_dataset')
    print(e.embedding(content = 'hello, world'))
