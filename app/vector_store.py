# app/vector_store.py

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from app.utils import chunk_text

class VectorStore:
    def __init__(self, index_path="index/index.faiss", meta_path="index/index.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)  # 384-dim model
        self.metadata = []  # stores (source_path, chunk_text)

    def add_documents(self, documents: list[tuple[str, str]]):
        all_chunks = []
        chunk_sources = []
        for filepath, text in documents:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            chunk_sources.extend([(filepath, chunk) for chunk in chunks])

        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.index.add(embeddings)
        self.metadata.extend(chunk_sources)

        self.save()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, k: int = 5):
        query_emb = self.model.encode([query])
        distances, indices = self.index.search(query_emb, k)
        return [self.metadata[i] for i in indices[0]]
