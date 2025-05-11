from langchain.schema.messages import HumanMessage, SystemMessage
from app.vector_store import VectorStore
from langchain_openai import ChatOpenAI

class RAGPipeline:
    def __init__(self,model_name="deepseek/deepseek-chat-v3-0324:free"):
        print("Initializing ChatOpenAI model...")
        self.llm = ChatOpenAI(
            temperature=0.2,
            model_name=model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key="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",
        )

        print("Loading vector store...")
        self.vs = VectorStore()
        self.vs.load()

    def run(self, query: str, k: int = 5):
        # Step 1: Retrieve top-k docs
        docs = self.vs.search(query, k=k)
        context = "\n".join([f"[{i+1}] {chunk}" for i, (_, chunk) in enumerate(docs)])

        # Step 2: Construct prompt
        messages = [
            SystemMessage(
                content="You are a helpful assistant. Answer the question with explanation based strictly on the provided context. Give code whenever query asks for it."
            ),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
        ]

        # Step 3: Generate response
        response = self.llm.invoke(messages)
        return {
            "answer": response.content.strip(),
            "source_files": list({path for path, _ in docs})
        }
