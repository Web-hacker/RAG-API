import requests
import os
import json
from pathlib import Path
from typing import List
from app.vectore_store import VectorStore
from app.utils import hybrid_pdf_extraction,extract_text_from_image

COMMIT_FILE = "data/last_commit.json"


def save_last_commit(repo: str, commit_sha: str):
    with open(COMMIT_FILE, 'w') as f:
        json.dump({"repo": repo, "commit": commit_sha}, f)


def load_last_commit() -> dict:
    if os.path.exists(COMMIT_FILE):
        with open(COMMIT_FILE, 'r') as f:
            return json.load(f)
    return {}


def get_latest_commit_sha(owner: str, repo: str, branch: str = "main") -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    res = requests.get(url)
    res.raise_for_status()
    return res.json()["sha"]


def get_changed_files(owner: str, repo: str, base_sha: str, head_sha: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}"
    res = requests.get(url)
    res.raise_for_status()
    files = res.json().get("files", [])
    changed = []
    deleted = []
    for f in files:
        if f["status"] in {"modified", "added"}:
            changed.append(f["filename"])
        elif f["status"] == "removed":
            deleted.append(f["filename"])
    return changed, deleted


def ingest_changed_files(repo_url: str, branch: str = "main"):
    parts = repo_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]

    latest_sha = get_latest_commit_sha(owner, repo, branch)
    state = load_last_commit()

    if state.get("repo") == repo and state.get("commit") == latest_sha:
        print("No new commit. Skipping ingestion.")
        return

    vs = VectorStore()
    
    if state.get("repo") == repo:
        changed_files, deleted_files = get_changed_files(owner, repo, state["commit"], latest_sha)
    else:
        # First time: fetch all files from HEAD
        tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        tree_res = requests.get(tree_url)
        tree_res.raise_for_status()
        changed_files = [f["path"] for f in tree_res.json().get("tree", []) if f["type"] == "blob"]
        deleted_files = []
        # print(changed_files)

    # Handle deleted files
    for path in deleted_files:
        print(f"Removing deleted file from vector DB: {path}")
        vs.remove_document(path)

    # Handle added/modified files
    for path in changed_files:
        
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        content_res = requests.get(raw_url)
        # if(path.suffix().lower()==):
        # ext=Path(path).suffix().lower()
        ext = path[path.rfind('.'):].lower() if '.' in path else ''
        # print(content_res.status_code)
        print(ext)
        if ext== ".pdf":
            # print(content_res.text)
            content_res.text=hybrid_pdf_extraction(content_res.content)
        elif ext in {".png", ".jpg", ".jpeg", ".gif" }:
            continue
        #     print(content_res.content)
        #     content_res.text=extract_text_from_image(content_res.content)

        # print(content_res.text)
        if content_res.status_code == 200 and len(content_res.text):
            print(f"Re-ingesting file: {path}")
            vs.remove_document(path)  # Remove old embedding if exists
            # print(10)
            vs.upsert(doc_id=path, content=content_res.text)
        else:
            print(f"Failed to fetch content for: {path} (status {content_res.status_code})")

    save_last_commit(repo, latest_sha)
