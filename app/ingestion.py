# app/ingestion.py

import os
from git import Repo
from app.utils import load_text_files_from_dir, load_files_from_file_path
from app.vector_store import VectorStore

def clone_repo(repo_url: str, branch: str = "main", save_dir: str = "data") -> str:
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.join(save_dir, repo_name)

    if os.path.exists(local_path):
        print(f"Repo already exists at {local_path}. Skipping clone.")
        return local_path

    print(f"Cloning {repo_url} into {local_path}...")
    Repo.clone_from(repo_url, local_path, branch=branch)
    return local_path


def ingest_repo_to_vector_db(repo_path: str):
    print(f"Ingesting docs from: {repo_path}")
    docs = load_text_files_from_dir(repo_path)
    vs = VectorStore()
    vs.add_documents(docs)
    print(f"Indexed {len(docs)} documents.")

def ingest_file_to_vector_db(file_path : str):
    print(f"Ingesting file from: {file_path}")
    docs = load_files_from_file_path(file_path)
    vs = VectorStore()
    vs.add_documents(docs)
    print(f"Indexed {len(docs)} documents.")