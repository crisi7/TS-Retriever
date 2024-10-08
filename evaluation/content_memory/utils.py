from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import json

class Singleton:
    def __init__(self, cls):
        self._cls = cls
    def instance(self, **kwargs):
        try:
            return self._instance
        except:
            self._instance = self._cls(**kwargs)
            return self._instance

def load_config(config_path):
    def wrapper(cls):
        cls.config = json.load(config_path)
        return cls
    return wrapper

def load_pipeline(modelpath):
    pipe = pipeline("text-generation", model=modelpath, torch_dtype=torch.bfloat16, device_map="auto")
    return pipe

def load_model(modelpath, local_files_only=False, trust_remote_code=False):
    model = AutoModel.from_pretrained(modelpath, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
    return model

def load_tokenizer(modelpath):
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    return tokenizer