from utils import load_experiment_doc, load_query_file, get_metric_at_k, load_embed_model_by_name, save_json, read_json
from content_memory import MemoryBuffer, MemoryDoc, ContrieverEmbedder
from pathlib import Path
import json
import numpy as np
import os
from tqdm import tqdm
import fire

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
import numpy as np

def eval_result(data):
    data_len = len(data)
    k_range = [1, 3, 5, 10, 20, 50, 100]
    result_recall = {
        "Mar@1": 0,
        "Mar@3": 0,
        "Mar@5": 0,
        "Mar@10": 0,
        "Mar@20": 0,
        "Mar@50": 0,
        "Mar@100":0
    }

    result_precision = {
        "Map@1": 0,
        "Map@3": 0,
        "Map@5": 0,
        "Map@10": 0,
        "Map@20": 0,
        "Map@50": 0,
        "Map@100":0
    }
    for d in data:
        metric = d['metrics_at_k']
        for k in k_range:
            result_precision[f'Map@{k}'] += metric[k]['p']
            result_recall[f'Mar@{k}'] += metric[k]['r']
    for k in k_range:
        p = result_precision[f'Map@{k}'] / data_len * 100
        r = result_recall[f'Mar@{k}'] / data_len * 100
        print(f"map@{k}: {p}")
        print(f"mar@{k}: {r}")

def experiment_nobel_with_dense_embed_model(doc_embed_path, query_file, embed_model_name, save_dir_path):
    result_data = []
    embedder = load_embed_model_by_name(embed_model_name)
    memory_buffer = load_experiment_doc(save_dir_path=doc_embed_path)
    queries = load_query_file(query_file)
    for d in tqdm(queries):
        if 'embedding' in d.keys():
            query_embedding = np.array(d['embedding'])
        else:
            query_embedding = embedder.embedding(content = d['query'], return_type = 'nparray')
        query_doc = MemoryDoc()
        query_doc.content = d['query']
        query_doc.embedding = query_embedding
        matchs = memory_buffer.similarity_retrieval_memory(
            simi_query=query_doc,
            top_k=100,
            thresh_hold=0.01
        )
        predict_items = [match.content for match in matchs]
        predict_names = [item.split('\n')[0] for item in predict_items]
        true_items = d['positive_text']
        answer_names = [item.split('\n')[0] for item in true_items]
        metrics_at_k = {}
        for k in [1,3,5,10,20,50,100]:
            p, r = get_metric_at_k(k, true_items, predict_items)
            metrics_at_k[k] = {'p': p, 'r': r}
        new_d = {
            'query': d['query'],
            'nums_relevant_doc': len(true_items),
            'metrics_at_k': metrics_at_k,
            'answer_name': answer_names,
            'predict_name': predict_names
        }
        result_data.append(new_d) 
    savepath = save_dir_path + f'/result_{embed_model_name}_name_nobel.json'
    save_json(result_data, savepath)
    eval_result(result_data)

def simple_tok(sent:str):
    return sent.lower().split()

def experiment_nobel_with_bm25(doc_path:str, query_file:str, embed_model_name:str, save_dir_path:str):
    result_data = []
    queries = load_query_file(query_file)
    corpus = read_json(doc_path)
    
    tok_corpus = [simple_tok(s) for s in corpus]
    dictionary = Dictionary(tok_corpus)
    bm25_model = OkapiBM25Model(dictionary=dictionary)
    bm25_corpus = bm25_model[list(map(dictionary.doc2bow, tok_corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(tok_corpus), num_terms=len(dictionary),normalize_queries=False, normalize_documents=False)
    tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')


    for d in tqdm(queries):
        query = simple_tok(d['query'])
        tfidf_query = tfidf_model[dictionary.doc2bow(query)]
        similarities = bm25_index[tfidf_query]
        top_100_indices_descending = np.argsort(similarities)[-100:][::-1]
        predict_items = [corpus[i] for i in top_100_indices_descending]
        predict_names = [item.split('\n')[0] for item in predict_items]
        try:
            true_items = d['positive_text']
        except:
            true_items = d['passages']
        answer_names = [item.split('\n')[0] for item in true_items]
        metrics_at_k = {}
        for k in [1,3,5,10,20,50,100]:
            p, r = get_metric_at_k(k, true_items, predict_items)
            metrics_at_k[k] = {'p': p, 'r': r}
        new_d = {
            'query': d['query'],
            '_id': d['_id'],
            'nums_relevant_doc': len(true_items),
            'metrics_at_k': metrics_at_k,
            'answer_name': answer_names,
            'predict_name': predict_names
        }
        result_data.append(new_d) 
    savepath = save_dir_path + f'/result_{embed_model_name}_name_nobel.json'
    save_json(result_data, savepath)
    eval_result(result_data)

def experiment_with_bm25():
    embed_model_name = 'bm25'
    doc_path = './data/nobel_prize/doc.json'
    query_file = './data/nobel_prize/query.json'
    save_dir_path = './temp_embed_files/docs/bm25_embed_doc'
    experiment_nobel_with_bm25(
        doc_path=doc_path,
        query_file=query_file,
        embed_model_name = embed_model_name,
        save_dir_path = save_dir_path
    )

def experiment(embed_model_name_query, embed_model_name_doc, query_embed_dir, doc_embed_dir):
    if embed_model_name_query == 'bm25':
        experiment_with_bm25()
    else:
        save_dir_path = doc_embed_dir
        query_file =  Path(query_embed_dir) / f"{embed_model_name_query}_query_embbed.json"
        experiment_nobel_with_dense_embed_model(doc_embed_dir, query_file, embed_model_name_query, save_dir_path)

if __name__ == '__main__':
    fire.Fire(experiment)