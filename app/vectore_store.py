"""
VectorStore class using FAISS (IndexFlatL2 + IndexIDMap) and LangChain GoogleGenerativeAIEmbeddings.
Supports efficient upsertion and retrieval of small to medium document collections.
"""

import faiss
import pickle
import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.utils import chunk_text

#env_path = Path(__file__).parent / ".env"
#load_dotenv(dotenv_path=env_path)

class VectorStore:
    def __init__(
        self,
        index_path="index/index.faiss",
        meta_path="index/metadata.pkl"
    ):

        print("VectorStore: Initializing...")
        #api_key = os.getenv("GOOGLE_API_KEY")
        #if not api_key:
        #    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")
        #os.environ["GOOGLE_API_KEY"] = api_key

        self.index_path = index_path
        self.meta_path = meta_path
        print("Loading embedding model...")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        print("Embedding model loaded.")
        #self.model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        self.index = None
        self.metadata: Dict[str, dict] = {}
        self.doc_id_by_faiss_id: Dict[int, str] = {}

        self._load_or_initialize_index()

    def _load_or_initialize_index(self):
     try:
        if os.path.exists(self.index_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
        else:
            print("Creating new FAISS index...")
            sample_vec = self.model.encode(["sample"], convert_to_numpy=True)
            dim = len(sample_vec[0])
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

        if os.path.exists(self.meta_path):
            print("Loading metadata...")
            with open(self.meta_path, "rb") as f:
                data = pickle.load(f)
                self.metadata = data.get("metadata", {})
                self.doc_id_by_faiss_id = data.get("faiss_id_map", {})

     except Exception as e:
        print(f"[ERROR] Failed to initialize FAISS index: {e}")


    def get_file_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(block)
        return sha256_hash.hexdigest()

    def _generate_faiss_ids(self, count: int) -> List[int]:
        base = self.index.ntotal
        return list(range(base, base + count))

    def upsert(self, doc_id: str, file_path: str, content: str):
        content_hash = self.get_file_hash(file_path)

        if doc_id in self.metadata and self.metadata[doc_id]["content_hash"] == content_hash:
            print(f"[{doc_id}] Skipping: No changes.")
            return
        elif doc_id in self.metadata:
            print(f"[{doc_id}] Updating existing document...")
        else:
            print(f"[{doc_id}] Adding new document...")

        chunks = chunk_text(content)
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        faiss_ids = self._generate_faiss_ids(len(embeddings))

        self.index.add_with_ids(np.array(embeddings, dtype="float32"), np.array(faiss_ids))
        for fid in faiss_ids:
            self.doc_id_by_faiss_id[fid] = doc_id

        self.metadata[doc_id] = {
            "content_hash": content_hash,
            "content": content,
            "chunks": chunks,
            "faiss_ids": faiss_ids
        }

        self.save()
        print(f"[{doc_id}] Upsert complete.")

    def upsert_documents(self, documents: List[Tuple[str, str]]):
        for file_path, content in documents:
            doc_id = file_path  # You can change this to any unique ID
            self.upsert(doc_id, file_path, content)

    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        query_vec = self.model.encode([query], convert_to_numpy=True)[0]
        D, I = self.index.search(np.array([query_vec], dtype="float32"), k)

        results = []
        for idx in I[0]:
            if idx == -1:
                continue
            doc_id = self.doc_id_by_faiss_id.get(idx)
            if not doc_id or doc_id not in self.metadata:
                continue
            doc_meta = self.metadata[doc_id]
            results.append({
                "doc_id": doc_id,
                "content": doc_meta["content"],
                "chunks": doc_meta["chunks"]
            })
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "faiss_id_map": self.doc_id_by_faiss_id
            }, f)
