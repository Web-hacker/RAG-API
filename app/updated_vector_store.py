"""
This module defines the VectorStore class for managing document embeddings using FAISS
and SentenceTransformers. It supports upserting documents (with change tracking via hashing),
chunking content, storing vector-document mappings, and performing similarity searches.

Dependencies:
- faiss
- pickle
- sentence_transformers
- hashlib
- os
- app.utils (for chunk_text function)

"""

import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.utils import chunk_text
import os
import hashlib
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class VectorStore:
    """
    A vector database manager using FAISS for similarity search and
    SentenceTransformer for embedding documents.

    Attributes:
        index_path (str): Path to save/load FAISS index.
        meta_path (str): Path to save/load document metadata.
        vector_map_path (str): Path to save/load doc_id-to-vector mapping.
        model (SentenceTransformer): Sentence embedding model.
        index (faiss.Index): FAISS index storing the vector embeddings.
        metadata (dict): Metadata about documents, including hash and content.
        doc_id_by_vector_idx (list): Mapping from vector index to document ID.

        reserve_model = all-MiniLM-L6-v2
    """

    def __init__(self, index_path="index/index.faiss", meta_path="index/index.pkl", vector_map_path="index/vector_map.pkl"):
        """
        Initialize the vector store by loading the FAISS index and metadata if available.
        
        """

        #api_key = os.getenv("GOOGLE_API_KEY")
        #if not api_key:
        #    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")

        #os.environ["GOOGLE_API_KEY"] = api_key

        self.index_path = index_path
        self.meta_path = meta_path
        self.vector_map_path = vector_map_path
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        #self.model = GoogleGenerativeAIEmbeddings(model="embedding-001")
        self.index = faiss.IndexFlatL2(1024)
        self.metadata = {}
        self.doc_id_by_vector_idx = []

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        if os.path.exists(self.vector_map_path):
            with open(self.vector_map_path, "rb") as f:
                self.doc_id_by_vector_idx = pickle.load(f)

    def get_file_hash(self, file_path: str) -> str:
        """
        Compute the SHA-256 hash of the file's contents.

        Args:
            file_path (str): Absolute or relative path to the file.

        Returns:
            str: SHA-256 hex digest of the file.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def upsert(self, doc_id: str, file_path: str, content: str):
        """
        Insert or update a document based on its hash. Avoids re-embedding if unchanged.

        Args:
            doc_id (str): Unique identifier for the document (typically the file path).
            file_path (str): Path to the source file.
            content (str): Text content of the document.
        """
        file_hash = self.get_file_hash(file_path)

        if doc_id in self.metadata:
            stored_hash = self.metadata[doc_id]["content_hash"]
            if stored_hash == file_hash:
                print(f"Document {doc_id} has not changed. Skipping re-embedding.")
                return
            else:
                print(f"Document {doc_id} has changed. Re-embedding.")
        else:
            print(f"Adding new document {doc_id} to the index.")

        chunks = chunk_text(content)
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        idx = self.index.ntotal
        self.index.add(embeddings)
        self.doc_id_by_vector_idx.extend([doc_id] * len(embeddings))

        self.metadata[doc_id] = {
            "index": idx,
            "content_hash": file_hash,
            "content": content,
            "chunks": chunks
        }
        self.save()
        print(f"Complete {doc_id}")

    def upsert_documents(self, documents: list[tuple[str, str]]):
        """
        Insert or update multiple documents into the index.

        Args:
            documents (list of tuples): Each tuple is (file_path, content).
        """
        for file_path, content in documents:
            doc_id = file_path
            self.upsert(doc_id, file_path, content)

    def save(self):
        """
        Persist the FAISS index, metadata, and document-vector mapping to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(self.vector_map_path, "wb") as f:
            pickle.dump(self.doc_id_by_vector_idx, f)

    def load(self):
        """
        Load the FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(self.index_path)
        self.load_metadata()
        if os.path.exists(self.vector_map_path):
            with open(self.vector_map_path, "rb") as f:
                self.doc_id_by_vector_idx = pickle.load(f)

    def load_metadata(self):
        """
        Load document metadata from the metadata pickle file.
        """
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, k: int = 5):
        """
        Perform vector similarity search for a given query.

        Args:
            query (str): Natural language search query.
            k (int): Number of top documents to retrieve.

        Returns:
            list of dict: Top-k documents with their IDs, content, and chunks.
        """
        query_emb = self.model.encode([query])
        distances, indices = self.index.search(query_emb, k)
        results = []
        for idx in indices[0]:
            if idx < len(self.doc_id_by_vector_idx):
                doc_id = self.doc_id_by_vector_idx[idx]
                if doc_id in self.metadata:
                    results.append({
                        "doc_id": doc_id,
                        "content": self.metadata[doc_id]["content"],
                        "chunks": self.metadata[doc_id]["chunks"]
                    })
        return results
