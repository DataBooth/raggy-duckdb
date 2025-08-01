import os

import numpy as np
import ollama
from dotenv import load_dotenv
from together import Together


# Embedding backend base class
class EmbeddingBackend:
    def embed(self, texts):
        raise NotImplementedError


class TogetherEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model):
        load_dotenv()  # Reads .env file if present
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("Set TOGETHER_API_KEY in your environment or .env file.")
        self.client = Together(api_key=api_key)
        self.model = model

    def embed(self, texts):
        result = self.client.embeddings.create(model=self.model, input=texts)
        return [np.array(e.embedding, dtype=np.float32) for e in result.data]


class OllamaEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model):
        self.model = model

    def embed(self, texts):
        results = []
        for text in texts:
            out = ollama.embeddings(model=self.model, prompt=text)
            embed = out.get("embedding")
            results.append(np.array(embed, dtype=np.float32))
        return results


# LLM backend base class
class LLMBackend:
    def chat(self, messages):
        raise NotImplementedError


class TogetherLLMBackend(LLMBackend):
    def __init__(self, model):
        load_dotenv()  # Reads .env file if present
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("Set TOGETHER_API_KEY in your environment or .env file.")
        self.client = Together(api_key=api_key)
        self.model = model

    def chat(self, messages):
        result = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return result.choices[0].message.content


class OllamaLLMBackend(LLMBackend):
    def __init__(self, model):
        self.model = model

    def chat(self, messages):
        out = ollama.chat(model=self.model, messages=messages)
        return out["message"]["content"]


# Backends selection functions
def create_embedding(provider, model):
    if provider == "together":
        return TogetherEmbeddingBackend(model)
    elif provider == "ollama":
        return OllamaEmbeddingBackend(model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def create_llm(provider, model):
    if provider == "together":
        return TogetherLLMBackend(model)
    elif provider == "ollama":
        return OllamaLLMBackend(model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
