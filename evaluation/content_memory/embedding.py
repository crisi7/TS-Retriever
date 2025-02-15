from abc import ABC, abstractmethod
import requests
import openai
import numpy as np
import json
import time
import random
from .utils import Singleton, load_model, load_tokenizer, load_pipeline, load_config
from peft import PeftConfig, PeftModel
import torch
import torch.nn as nn

class Embedder(ABC):
    @abstractmethod
    def embedding(self, content: str):
        """ Give contents and return summarization
            Args:
                contents: The content list to input
            Return:
                The summrization from content
        """
        pass


class OpenaiEmbedder(Embedder):
    def __init__(self,api_base=None, api_key=None):
        self.token_limit = 8191
        if api_base is None or api_key is None:
            raise ValueError(f"Provide api_base(api_url) and api_key if use OpenaiEmbedder")
        self.api_base = api_base
        self.api_key = api_key
    def _embedding(self, content, model="text-embedding-ada-002",return_type='nparray'):
        """
        转接供应商的OpenAI服务，收费的，不要梯子，支持全部OpenAI模型
        默认的gpt-3.5-turbo
        :param prompt:
        :param model: 可选的模型包括：'gpt-3.5-turbo','gpt-3.5-turbo-16k','gpt-3.5-turbo-instruct','gpt-4','gpt-4-32k','text-davinci-003'
        :return:
        """
        proxies = {
            "http": None,
            "https": None,
        }
        #url = "https://api.closeai-proxy.xyz/v1/embeddings"
        url = self.api_base
        headers = {
            "Content-Type": "application/json",
            'Authorization': api_key,
        }
        data = {
            "model": model,
            "input": content
        }
        response = requests.request("POST", url, headers=headers, data=json.dumps(data), proxies=proxies)
        # print(response)
        res = response.json()
        if return_type == 'nparray':
            return np.array(res['data'][0]['embedding'])
        else:
            return res['data'][0]['embedding']
    def embedding(self, content, model="text-embedding-ada-002",return_type='nparray'):
        try:
            return self._embedding(content, model=model, return_type=return_type)
        except:
            time.sleep(10)
            return self._embedding(content, model=model, return_type=return_type)

class JinaaiEmbedder(Embedder):
    def __init__(self, model_name_or_path):
        self.model = load_model(model_name_or_path, local_files_only=False, trust_remote_code=True)
    def embedding(self, content, return_type = 'nparray'):
        content_list = [content]
        embeddings = self.model.encode(content_list)
        if return_type == 'nparray':
            return embeddings[0]
        else:
            return embeddings[0].tolist()
    def batch_embedding(self, contents, return_type='json'):
        embeddings = self.model.encode(contents)
        if return_type == 'json':
            return [embedding.tolist() for embedding in embeddings]
        elif return_type == 'nparray':
            return np.array(embeddings)
        else:
            raise ValueError(f"Unsupported return type: {return_type}")

class TasbEmbedder(Embedder):
    def __init__(self, model_name_or_path):
        self.model = load_model(model_name_or_path, local_files_only=False)
        self.tokenizer = load_tokenizer(model_name_or_path)
    def embedding(self, content, return_type = 'nparray'):
        inputs = self.tokenizer(content, return_tensors='pt')
        embeddings = self.model(**inputs)[0][:,0,:].squeeze(0)
        if return_type == 'nparray':
            return embeddings.detach().numpy()
        else:
            return embeddings.detach().numpy().tolist()
    def batch_embedding(self, contents, return_type='json'):
        all_embeddings = []
        for content in contents:
            embedding = self.embedding(content, return_type=return_type)
            all_embeddings.append(embedding)
        return all_embeddings

class ContrieverEmbedder(Embedder):
    def __init__(self, model_name_or_path):
        self.model = load_model(model_name_or_path)
        self.tokenizer = load_tokenizer(model_name_or_path)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def batch_embedding(self, contents, return_type = 'nparray'):
        inputs = self.tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        if return_type == 'nparray':
            return embeddings.detach().numpy()
        else:
            return embeddings.detach().numpy().tolist()

    def embedding(self, content, return_type = 'nparray'):
        return self.batch_embedding([content], return_type=return_type)[0]

class LoraContrieverEmbedder(Embedder):
    def __init__(self, model_name_or_path=None, lora_model_path=None):
        model = load_model(model_name_or_path)
        self.tokenizer = load_tokenizer(model_name_or_path)
        self.model = PeftModel.from_pretrained(model, lora_model_path)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def batch_embedding(self, contents, return_type = 'nparray'):
        inputs = self.tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        if return_type == 'nparray':
            return embeddings.detach().numpy()
        else:
            return embeddings.detach().numpy().tolist()
    
    def embedding(self, content, return_type = 'nparray'):
        return self.batch_embedding([content], return_type=return_type)[0]

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class RouterDualEncoderRetriever(Embedder):
    def __init__(self, query_encoder=None, passage_encoder=None, router_state_dict_path=None):
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        router = Classifier(input_dim=768, hidden_dim=128, output_dim=1)
        router.load_state_dict(torch.load(router_state_dict_path))
        self.router = router
    
    def embedding(self, content: str, return_type = 'nparray'):
        embed = torch.tensor(self.passage_encoder.embedding(content=content))
        self.router.eval()
        with torch.no_grad():
            outputs = self.router(embed)
            predicted = (outputs.squeeze() > 0.5).float()
        query_embed = predicted * embed + (1-predicted) * self.query_encoder.embedding(content)
        if return_type == 'nparray':
            return query_embed
        else:
            return query_embed.tolist()

    def batch_embedding(self, contents, return_type='nparry'):
        embeds = []
        for content in contents:
            embed = self.embedding(content, return_type=return_type)
            embeds.append(embed)
        if return_type == 'nparray':
            try:
                import numpy as np
                return np.array(embeds)
            except ImportError:
                raise ImportError("Numpy is required to return embeddings as numpy array. Please install it.")
        else:
            return embeds


if __name__ == "__main__":
    encoder = TasbEmbedder()
    encoder = e.embedding(content = 'hello')
