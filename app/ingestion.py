
import os
import shutil
import stat
from git import Repo

from app.utils import load_text_files_from_dir, load_files_from_file_path
from app.updated_vector_store import VectorStore

def handle_remove_readonly(func, path, exc_info):
    """
    Force deletes read-only files and directories (e.g., .git) during rmtree.
    Called via `shutil.rmtree(..., onerror=handle_remove_readonly)`
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repo(repo_url: str, branch: str = "main", save_dir: str = "data") -> str:
    """
    Clones a Git repository from the specified URL and branch to the given save directory.
    Removes existing repo folder if it already exists.

    Args:
        repo_url (str): URL of the GitHub repository.
        branch (str): Branch to clone (default: 'main').
        save_dir (str): Target directory to save the repo (default: 'data').

    Returns:
        str: Local path to the cloned repository.
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.join(save_dir, repo_name)

    if os.path.exists(local_path):
        shutil.rmtree(local_path, onerror=handle_remove_readonly)

    print(f"[INFO] Cloning repo: {repo_url} -> {local_path}")
    Repo.clone_from(repo_url, local_path, branch=branch)
    return local_path

def ingest_repo_to_vector_db(repo_path: str):
    """
    Loads and ingests all text files from a cloned repository into the vector store.

    Args:
        repo_path (str): Path to the local cloned repository.
    """
    print(f"[INFO] Ingesting repository documents from: {repo_path}")
    docs = load_text_files_from_dir(repo_path)
    print("Loaded Docs")
    vs = VectorStore()
    print("Loaded Vectorstore")
    vs.upsert_documents(docs)
    print(f"[SUCCESS] Indexed {len(docs)} documents from repository.")

def ingest_file_to_vector_db(file_path: str):
    """
    Loads and ingests a single uploaded file into the vector store.

    Args:
        file_path (str): Full file path to the uploaded document.
    """
    print(f"[INFO] Ingesting single file from: {file_path}")
    docs = load_files_from_file_path(file_path)
    vs = VectorStore()
    vs.upsert_documents(docs)
    print(f"[SUCCESS] Indexed {len(docs)} documents from file.")
