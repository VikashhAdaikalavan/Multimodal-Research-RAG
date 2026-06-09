import os
from langchain_chroma import Chroma
import gc

class Retriever:
    def __init__(self):
        self.vector_store = None

    def load_vector_store(self, embedding_model):
        self.vector_store = Chroma(
            persist_directory=os.getenv("DB_PATH", r"D:\DRDO PROJECT\RAG Assistant\ChromaDB"),
            embedding_function=embedding_model
        )

    def retrieve_documents(self, query: str, top_k: int = 10):
        return self.vector_store.similarity_search_with_score(
            query=query, k=top_k
        )

    def filter_relevant_documents(self, docs, threshold: float = 0.85):
        return [(doc, score) for doc, score in docs if score < threshold]
    
    def release(self):
        if self.vector_store is not None:
            try:
                self.vector_store._client.close()
            except Exception:
                pass
            self.vector_store = None

    def close(self):
        self.vector_store = None
        gc.collect()

    def build_prompt(self, query: str, docs: list) -> str:
        context_blocks = ""
        for doc, score in docs:
            context_blocks += (
                f"\n[Score: {score:.4f}]\n"
                f"{doc.page_content}\n"
                f"(source: {doc.metadata})\n"
                f"{'─'*40}\n"
            )

        return f"""You are a precise, knowledgeable research assistant specialising in drone technology.

                RULES — follow these without exception:
                1. Every factual claim must be traceable to the context below.
                2. Do NOT add background knowledge, assumptions, or definitions that are absent from the context.
                3. If the context does not cover the question, say exactly:
                "I couldn't find that in the uploaded documents."
                4. Never infer meanings of symbols or variables unless the context defines them.
                5. Combine information from all relevant chunks into one coherent answer.
                6. Be concise yet thorough; use bullet points or numbered lists where they aid clarity.
                7. Output only the final answer — no meta-commentary or thinking aloud.
                8. For image queries: rely on the context chunks; use the image description only to frame the question.

                CONTEXT:
                {context_blocks}

                QUESTION:
                {query}

                ANSWER:"""