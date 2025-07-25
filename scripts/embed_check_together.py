import os
import numpy as np
import diskcache as dc
from together import Together
from dotenv import load_dotenv

load_dotenv()
cache = dc.Cache("./cache_dir")


class TogetherAIEmbedder:
    def __init__(self, model: str):
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY must be set in environment")
        self.client = Together(api_key=api_key)
        self.model = model

    def embed(self, texts):
        results = []
        texts_to_call = []
        indices_to_call = []

        for i, text in enumerate(texts):
            cached_emb = cache.get(text)
            if cached_emb is not None:
                results.append(cached_emb)
            else:
                results.append(None)
                texts_to_call.append(text)
                indices_to_call.append(i)

        if texts_to_call:
            api_embeddings = self.client.embeddings.create(
                model=self.model, input=texts_to_call
            )
            # Extract actual vectors from embedding objects
            api_embs = [
                np.array(e.embedding, dtype=np.float32) for e in api_embeddings.data
            ]
            for idx, emb in zip(indices_to_call, api_embs):
                cache[texts_to_call[idx]] = emb
                results[idx] = emb
        return results


def test_embedding_call():
    try:
        model_name = "togethercomputer/m2-bert-80M-32k-retrieval"  # example valid embedding model
        embedder = TogetherAIEmbedder(model=model_name)
        test_text = ["Hello, this is a test embedding."]
        embeddings = embedder.embed(test_text)
        emb = embeddings[0]
        print(f"Successfully got embedding of shape: {emb.shape}")
    except Exception as e:
        print(f"Error during embedding call: {e}")


if __name__ == "__main__":
    test_embedding_call()
