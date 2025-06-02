
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.ingestion import clone_repo, ingest_repo_to_vector_db, ingest_file_to_vector_db
from app.rag import RAGPipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="GS RAG Agent", description="RAG Agent for file and repo ingestion", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/")
def root():
    return {"message": "GS RAG Agent is running!"}

# ----------------------------
# File Upload Endpoint
# ----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file and ingests its content into the vector database.
    """
    file_location = f"data/{file.filename}"
    
    # Remove old file if it exists
    if os.path.exists(file_location):
        os.remove(file_location)

    # Save the new file
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest into vector DB
    ingest_file_to_vector_db(file_location)

    return {"message": f"File '{file.filename}' saved and ingested successfully."}

# ----------------------------
# Repo Clone & Ingest Endpoint
# ----------------------------
class RepoInput(BaseModel):
    repo_url: str
    branch: str = "main"

@app.post("/clone")
def clone_repo_endpoint(input: RepoInput):
    """
    Clones a GitHub repo and ingests its contents into the vector database.
    """
    try:
        path = clone_repo(input.repo_url, input.branch)
        ingest_repo_to_vector_db(path)
        return {"message": f"Repo cloned and ingested successfully at '{path}'"}
    except Exception as e:
        return {"error": str(e)}

# ----------------------------
# RAG Query Endpoint
# ----------------------------
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def ask_question(request: QueryRequest):
    """
    Accepts a user query and returns an answer using the RAG pipeline.
    """
    rag = RAGPipeline()
    result = rag.run(request.query)
    
    return {
        "answer": result["answer"],
        "sources": result["source_files"]
    }
