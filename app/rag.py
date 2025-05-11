
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.updated_vector_store import VectorStore

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

class RAGPipeline:
    """
    A Retrieval-Augmented Generation (RAG) pipeline that:
    - Searches relevant context using vector DB
    - Constructs a prompt
    - Calls an LLM to generate responses based on retrieved context
    """

    def __init__(self, model_name: str = "deepseek/deepseek-chat-v3-0324:free"):
        """
        Initializes the LLM client and loads the vector store.
        """
        print("Initializing ChatOpenAI model...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

        self.llm = ChatOpenAI(
            temperature=0.2,
            model_name=model_name
        )

        print("Loading vector store...")
        self.vs = VectorStore()
        self.vs.load()

    def run(self, query: str, k: int = 5) -> dict:
        """
        Main RAG pipeline logic.

        Args:
            query (str): User query.
            k (int): Number of top documents to retrieve.

        Returns:
            dict: Contains final answer and source document IDs.
        """
        # Step 1: Retrieve top-k documents
        docs = self.vs.search(query, k=k)

        # Step 2: Format context
        context = "\n".join([f"[{i+1}] {doc['content']}" for i, doc in enumerate(docs)])

        # Step 3: Construct prompt
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Answer the question with a detailed explanation "
                    "based on the provided context. Include all important information and provide clean bash-formatted code where necessary."
                )
            ),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
        ]

        # Step 4: Generate answer
        response = self.llm.invoke(messages)

        return {
            "answer": response.content.strip(),
            "source_files": list({doc["doc_id"] for doc in docs})
        }
