import os
from dotenv import load_dotenv
from langchain.schema.messages import HumanMessage, SystemMessage
from app.updated_vector_store import VectorStore
from langchain_openai import ChatOpenAI
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class RAGPipeline:
    def __init__(self,model_name="deepseek/deepseek-chat-v3-0324:free"):
        print("Initializing ChatOpenAI model...")
        print("Loaded Key:", os.getenv("OPENAI_API_KEY"))
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        self.llm = ChatOpenAI(
            temperature=0.2,
            model_name=model_name,
        )

        print("Loading vector store...")
        self.vs = VectorStore()
        self.vs.load()

    def run(self, query: str, k: int = 5):
        # Step 1: Retrieve top-k docs
        docs = self.vs.search(query, k=k)
        print(docs)
        context = "\n".join([f"[{i+1}] {body['content']}" for i, body in enumerate(docs)])

        # Step 2: Construct prompt
        messages = [
            SystemMessage(
                content="You are a helpful assistant. Answer the question with detailed explanation based  on the provided context. Make complete use of provided context. Give every vital information from context in detailed manner. Give code whenever necessary in clean bash format."
            ),
            HumanMessage(
                content=f"Context:\n{context} \n\nQuestion: {query}\nAnswer:"
            )
        ]

        # Step 3: Generate response
        response = self.llm.invoke(messages)
        return {
            "answer": response.content.strip(),
            "source_files": list({path["doc_id"] for path in docs})
        }
