embed_model_query="Tscontriever_with_differ_dataset"
embed_model_doc="Tscontriever_with_differ_dataset"
query_embed_save_dir="./temp_embed_files/query"
doc_embed_save_dir="./temp_embed_files/docs"

python embed_query.py \
    "${query_embed_save_dir}" \
    "./data/nobel_prize/query.json" \
    "${embed_model_query}"

python embed_doc.py \
    "${doc_embed_save_dir}/${embed_model_doc}_embed_doc" \
    "./data/nobel_prize/doc.json" \
    "${embed_model_doc}"

python experiment.py \
    "${embed_model_query}" \
    "${embed_model_doc}" \
    "${query_embed_save_dir}" \
    "${doc_embed_save_dir}/${embed_model_doc}_embed_doc"
