import faiss
import pickle
from sentence_transformers import SentenceTransformer
from app.utils import chunk_text  
import os
import hashlib

class VectorStore:
    def __init__(self, index_path="index/index.faiss", meta_path="index/index.pkl",vector_map_path="index/vector_map.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.vector_map_path = vector_map_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)  # 384-dim model
        
        # Load existing index and metadata if available
        self.metadata = {}
        self.doc_id_by_vector_idx = []

        # Load index and metadata if available
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
        Calculate SHA-256 hash for the content of the given file.
        
        Args:
        - file_path (str): Path to the file.
        
        Returns:
        - file_hash (str): SHA-256 hash of the file content.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def upsert(self, doc_id: str, file_path: str, content: str):
        """
        Upsert a document into the vector DB. It checks if the file's hash has changed before embedding.
        """
        file_hash = self.get_file_hash(file_path)

        # Check if the document already exists and if its hash has changed
        if doc_id in self.metadata:
            stored_hash = self.metadata[doc_id]["content_hash"]
            if stored_hash == file_hash:
                print(f"Document {doc_id} has not changed. Skipping re-embedding.")
                return
            else:
                print(f"Document {doc_id} has changed. Re-embedding.")
        else:
            print(f"Adding new document {doc_id} to the index.")

        # Chunk the content and create embeddings
        chunks = chunk_text(content)
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        # Add the embeddings to the FAISS index
        idx = self.index.ntotal  # The next available index position in FAISS
        self.index.add(embeddings)
        print(f"Complete {doc_id}")
        # Track which doc_id each chunk belongs to
        self.doc_id_by_vector_idx.extend([doc_id] * len(embeddings))
        print(f"Complete {doc_id}")
        # Update metadata (file path, content hash, and content)
        self.metadata[doc_id] = {
            "index": idx,
            "content_hash": file_hash,
            "content": content,
            "chunks": chunks
        }
        print(f"Complete {doc_id}")
        self.save()

    def save(self):
        """Save the FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(self.vector_map_path, "wb") as f:
            pickle.dump(self.doc_id_by_vector_idx, f)

    def load(self):
        """Load the FAISS index and metadata from disk."""
        self.index = faiss.read_index(self.index_path)
        self.load_metadata()
        if os.path.exists(self.vector_map_path):
            with open(self.vector_map_path, "rb") as f:
                self.doc_id_by_vector_idx = pickle.load(f)

    def load_metadata(self):
        """Load the document metadata (file path and content hash)."""
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, k: int = 5):
        """
        Perform a search using the query and return top-k results.
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
    
    def upsert_documents(self, documents: list[tuple[str, str]]):
        """
        Upsert multiple documents into the vector DB. This method checks the hash of each file to
        avoid re-embedding unchanged files.
        """
        for file_path, content in documents:
            doc_id = file_path  # Use the file path as the document ID
            self.upsert(doc_id, file_path, content)
