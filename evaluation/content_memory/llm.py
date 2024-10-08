from abc import ABC,abstractmethod
from typing import TypeVar, Dict
from .utils import load_pipeline, Singleton

import json
import torch

class LLM(ABC):    
    @abstractmethod
    def predict(self, prompt):
        raise NotImplementedError

@Singleton
class Zephyr(LLM):
    def __init__(self, model_name_or_path):
        self.pipeline = load_pipeline(model_name_or_path)
    def predict(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]
        input_prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(input_prompt, max_new_tokens=4096, do_sample=True, temperature=0.9, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"]

if __name__ == "__main__":
    llm = Zehpyr.instance()
    print(llm.predict('hi'))