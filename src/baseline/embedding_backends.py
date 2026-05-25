import requests
from sentence_transformers import SentenceTransformer


class OllamaEmbedder:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def encode(self, texts):
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model_name,
                    "input": text
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            # Ollama embed API returns "embeddings"
            if "embeddings" in data and len(data["embeddings"]) > 0:
                emb = data["embeddings"][0]
            elif "embedding" in data:
                emb = data["embedding"]
            else:
                raise ValueError(f"Unexpected Ollama response: {data}")

            embeddings.append(emb)

        return embeddings


class HFEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()


def get_embedder(backend: str, model_name: str):
    backend = backend.lower()

    if backend == "ollama":
        return OllamaEmbedder(model_name=model_name)

    if backend == "hf":
        return HFEmbedder(model_name=model_name)

    raise ValueError(f"Unsupported backend: {backend}")