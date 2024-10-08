from docarray import DocList, BaseDoc
from docarray.utils.filter import filter_docs
from docarray.typing import NdArray,ID
from typing import Union
from docarray.utils.find import find
import datetime
import uuid
import json
import os
from .embedding import OpenaiEmbedder

class MemoryDoc(BaseDoc):
    content: str = None
    embedding: NdArray = None
    doc_id: ID = None
    time: str = None

class MemoryBuffer:
    def __init__(self, embed_model=OpenaiEmbedder, file_path: str = None):
        self.docs = DocList[MemoryDoc]()
        self.embed_model = embed_model
        self.file_path = file_path if file_path is not None else './'
    
    def init_memory_node(self, content: str, time: str, embed_flag = True) -> MemoryDoc:
        new_doc = MemoryDoc()
        new_doc.content = content
        new_doc.doc_id = uuid.uuid4()
        new_doc.time = time if time is not None else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.embed_model is not None and embed_flag:
            new_doc.embedding = self.embed_model.embedding(content)
        return new_doc
    
    def add_memory(self, content: str, time:str) -> None:
        new_doc = self.init_memory_node(content,time)
        self.docs.append(new_doc)
    
    def similarity_retrieval_memory(self,
                                    simi_query: Union[str, MemoryDoc],
                                    top_k: int,
                                    thresh_hold:float,
                                    metric:str = 'cosine_sim',
                                    docs: DocList[MemoryDoc] = None) -> DocList[MemoryDoc]:
        index = self.docs if docs is None else docs
        if type(simi_query) is str:
            query = self.embed_model.embedding(simi_query)
        else:
            query = simi_query.embedding if simi_query.embedding is not None else self.embed_model(simi_query.content)
        top_matches, scores = find(
            index=index,
            query=query,
            limit=top_k,
            search_field='embedding',
            metric=metric,
        )
        thresh_hold_matches = DocList[MemoryDoc]()
        for i, match in enumerate(top_matches):
            if scores[i] >= thresh_hold:
                thresh_hold_matches.append(match)
            else:
                break
        return thresh_hold_matches

    def load_memory_from_binary(self, file_path: str = None) -> None:
        file_path = file_path if file_path is not None else self.file_path
        file_path_docs = file_path + "/doc"
        self.docs = self.docs.load_binary(file_path_docs, protocol='protobuf-array', compress=None, show_progress=False, streaming=False)
    
    def save_memory_to_binary(self, file_path: str = None) -> None:
        file_path = file_path if file_path is not None else self.file_path
        file_path_docs = file_path + "/doc"
        if not os.path.exists(file_path_docs):
            os.makedirs(os.path.dirname(file_path_docs), exist_ok=True)
        self.docs.save_binary(file_path_docs, protocol='protobuf-array', compress=None, show_progress=False)
