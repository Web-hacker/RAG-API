# app/api.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
from app.ingestion import clone_repo, ingest_repo_to_vector_db, ingest_file_to_vector_db
from app.rag import RAGPipeline

app = FastAPI()


@app.get("/")
def root():
    return {"message": "GS RAG Agent is running!"}

# Upload file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    ingest_file_to_vector_db(file_location)
    return {"message": f"File {file.filename} saved an dingested successfully to {file_location}"}

# New: Clone repo via API
class RepoInput(BaseModel):
    repo_url: str
    branch: str = "main"

@app.post("/clone")
def clone_repo_endpoint(input: RepoInput):
    try:
        path = clone_repo(input.repo_url, input.branch)
        ingest_repo_to_vector_db(path)
        return {"message": f"Repo cloned and ingested successfully to {path}"}
    except Exception as e:
        return {"error": str(e)}
    
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def ask_question(request: QueryRequest):
    rag = RAGPipeline()
    result = rag.run(request.query)
    return {
        "answer": result["answer"],
        "sources": result["source_files"]
    }
