import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


class RAGEngine:

    def __init__(
        self,
        vectorstore_path: str = os.path.join(
            "data", "vectorstore", "faiss_index"
        ),
    ):
        if not os.path.exists(vectorstore_path):
            raise FileNotFoundError(
                f"Vector store not found at {vectorstore_path}. Run 'python src/ingest.py' first!"
            )

        # 1. Load Local Hugging Face Embeddings & FAISS Index
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.load_local(
            vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # 2. Configure Retriever (Top-k = 3 most relevant chunks)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        # 3. Configure Groq LLM (Llama 3.1 8B)
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0)

        # 4. Strict Grounding System Prompt
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are an Enterprise AI Business Process Assistant.
Answer the user's question using ONLY the provided context excerpts from internal energy documents.

Strict Rules:
1. If the information required to answer the question is NOT in the context, respond explicitly with:
   "Information not available in policy documents."
2. Do NOT invent, assume, or extrapolate facts beyond the provided context.
3. Keep your response concise, structured, and professional.

Context:
{context}

Question: {question}

Answer:"""
        )

    def answer_question(self, query: str) -> dict:
        docs = self.retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        chain = self.prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"context": context_text, "question": query})

        return {
            "query": query,
            "answer": response,
            "sources": [
                {
                    "page": doc.metadata.get("page", 0) + 1,
                    "content": doc.page_content[:200] + "...",
                }
                for doc in docs
            ],
        }


if __name__ == "__main__":
    engine = RAGEngine()
    test_query = "What are the main energy sector projections or key points discussed?"
    result = engine.answer_question(test_query)

    print("\n--- TEST QUERY ---")
    print(f"Query: {result['query']}\n")
    print(f"Answer: {result['answer']}\n")
    print("--- RETRIEVED SOURCES ---")
    for i, src in enumerate(result["sources"], 1):
        print(f"[{i}] Page {src['page']}: {src['content']}\n")