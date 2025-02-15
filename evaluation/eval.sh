# 设置默认值
default_query="Tscontriever"
default_doc="Tscontriever"
default_qdir="./temp_embed_files/query"
default_ddir="./temp_embed_files/docs"

# 获取参数或使用默认值
embed_model_query=${1:-$default_query}
embed_model_doc=${2:-$default_doc}
query_embed_save_dir=${3:-$default_qdir}
doc_embed_save_dir=${4:-$default_ddir}

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
