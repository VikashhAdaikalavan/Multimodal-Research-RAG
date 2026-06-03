import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import time
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from Retriver import Retriever
from embedding_pipeline import EmbeddingModel, create_chunks
from Voice_input import VoiceInput
from Image_Input import ImageInput
from Speech_Output import Speaker
from vectordb import VectorDatabase
from parser import load_all_documents
import gc

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Terminal helpers
# ──────────────────────────────────────────────────────────────

def _typing(msg: str, delay: float = 0.03):
    for ch in msg:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def _banner():
    print("\n" + "─" * 54)
    print("RESEARCH ASSISTANT")
    print("─" * 54)

def _thinking_dots(label: str = "Thinking", n: int = 3):
    print(f"\n  {label}", end="", flush=True)
    for _ in range(n):
        time.sleep(0.4)
        print(".", end="", flush=True)
    print()

def _section(title: str):
    print(f"\n{'─'*54}")
    print(f"  {title}")
    print(f"{'─'*54}\n")

def _ok(msg: str):
    print(f"  ✓  {msg}")

def _warn(msg: str):
    print(f"  ⚠  {msg}")


# ──────────────────────────────────────────────────────────────
# Ingest — build / rebuild ChromaDB from scratch
# ──────────────────────────────────────────────────────────────

DB_PATH = r"D:\DRDO PROJECT\RAG Assistant\ChromaDB"

def run_ingest(embedding_model, retrieval: Retriever) -> None:
    """
    Wipes and rebuilds the vector database from the data folder,
    then hot-swaps the Retriever so the session uses the fresh
    index immediately — no restart needed.
    """
    _section("INGESTING DOCUMENTS")
    _typing("  Rebuilding the knowledge base from scratch.")
    _typing("  This may take a few minutes depending on file count.\n")

    try:
        # 1 — parse all PDFs / WAVs / images
        _thinking_dots("Loading documents from data folder")
        docs = load_all_documents()
        if not docs:
            _warn("No documents found in the data folder. Nothing to ingest.")
            return
        _ok(f"{len(docs)} document(s) loaded.")

        # 2 — chunk
        _thinking_dots("Splitting into chunks")
        chunks = create_chunks(docs)
        _ok(f"{len(chunks)} chunk(s) created.")

        # 3 — build ChromaDB
        _thinking_dots("Writing embeddings to ChromaDB")

        retrieval.release()
        retrieval.close()
        gc.collect()
        time.sleep(2)

        _thinking_dots("Writing embeddings to ChromaDB")
        vectordb = VectorDatabase()
        vectordb.create_vector_store(chunks, embedding_model)
        vectordb.close()
        del vectordb
        gc.collect()
        _ok("Vector database rebuilt successfully.")

        # 4 — hot-swap the retriever so no restart is needed
        retrieval.load_vector_store(embedding_model)
        _ok("Retriever updated — new knowledge base is live.\n")

        _typing("  All done! You can start asking questions right away.")

    except Exception as exc:
        _warn(f"Ingest failed: {exc}")
        _typing("The existing knowledge base is still active.")


# ──────────────────────────────────────────────────────────────
# Multi-model ensemble  +  judge
# ──────────────────────────────────────────────────────────────

AGENT_MODELS = ["mistral", "llama3", "qwen3:4b"]
JUDGE_MODEL  = "mistral"


class Chatbot:
    """
    Three Ollama agents each answer the RAG prompt independently.
    A judge model synthesises the single most accurate response.
    """

    def __init__(self):
        self.agents: dict[str, ChatOllama] = {}
        self.judge:  ChatOllama | None     = None
        self.llm:    ChatOllama | None     = None

    def load_llm(self):
        _typing("\nSpinning up the model ensemble…")
        for name in AGENT_MODELS:
            self.agents[name] = ChatOllama(model=name, temperature=0.1)
            _ok(f"Agent ready  →  {name}")
        self.judge = ChatOllama(model=JUDGE_MODEL, temperature=0.2)
        _ok(f"Judge  ready  →  {JUDGE_MODEL}\n")
        self.llm = list(self.agents.values())[0]

    def _query_agents(self, prompt: str) -> dict[str, str]:
        responses: dict[str, str] = {}
        for name, llm in self.agents.items():
            print(f"  ↳ asking {name}…", end=" ", flush=True)
            try:
                result = llm.invoke(prompt)
                responses[name] = result.content.strip()
                print("done")
            except Exception as exc:
                responses[name] = f"[{name} failed: {exc}]"
                print("failed")
        return responses

    @staticmethod
    def _judge_prompt(original_question: str, responses: dict[str, str]) -> str:
        block = "\n".join(
            f"── {model} ──\n{answer}\n"
            for model, answer in responses.items()
        )
        return f"""You are an expert evaluator for a drone-research Q&A system.

        ORIGINAL QUESTION:
        {original_question}

        MODEL RESPONSES:
        {block}

        YOUR TASK:
        1. Compare the three responses against each other.
        2. Penalise any claims not directly supported by the shared context.
        3. Pick the best supported facts from all three.
        4. Write ONE final, synthesised answer that is accurate, coherent, and concise.
        5. Use bullet points or numbered steps where they help clarity.
        6. Output the final answer only — no preamble, no "the best model was…" commentary.

        FINAL ANSWER:"""

    def generate_response(self, prompt: str, original_question: str = "") -> str:
        _thinking_dots("Consulting all three models")
        responses = self._query_agents(prompt)
        _thinking_dots("Judge is synthesising the best answer")
        judge_input = self._judge_prompt(
            original_question or prompt[:300],
            responses
        )
        try:
            result = self.judge.invoke(judge_input)
            return result.content.strip()
        except Exception as exc:
            _warn(f"Judge failed ({exc}), falling back to mistral's answer.")
            return responses.get("mistral", next(iter(responses.values())))


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _banner()
    _typing("\nHey! I'm your research assistant.\n"
        "Ask me anything about the uploaded research materials\n"
        "and I'll do my best to give you a solid, cited answer.\n")

    # ── Load embedding model ───────────────────────────────────
    print("Getting the embedding model ready…", end=" ", flush=True)
    embedding = EmbeddingModel()
    embedding.load_model()
    embedding_model = embedding.get_embedding_model()
    print("✓")

    # ── Load vector DB (auto-ingest on first run) ──────────────
    retrieval = Retriever()

    if not os.path.exists(DB_PATH):
        print("\n  No database found — running first-time ingest…")
        run_ingest(embedding_model, retrieval)
    else:
        print("Opening the knowledge base…", end=" ", flush=True)
        retrieval.load_vector_store(embedding_model)
        print("✓")

    # ── Load LLMs ──────────────────────────────────────────────
    chatbot = Chatbot()
    chatbot.load_llm()

    # ── Load voice / image / TTS ───────────────────────────────
    print("Warming up voice recognition…", end=" ", flush=True)
    voice = VoiceInput()
    voice.load_model()
    print("✓")

    image_handler = ImageInput()
    speaker       = Speaker()
    print("\nEverything's ready. Let's go!\n")

    # ── REPL ───────────────────────────────────────────────────
    while True:
        print("\n  How would you like to ask your question?")
        print("  [1] Type it out")
        print("  [2] Speak it")
        print("  [3] Show me an image")
        print("  [4] Re-ingest documents")
        print("  [5] Quit\n")

        choice = input("  → ").strip()

        # ── Input modes ────────────────────────────────────────
        if choice == "1":
            query = input("\n  Your question: ").strip()
            if not query:
                _warn("You didn't type anything — try again.")
                continue

        elif choice == "2":
            _typing("\n  Alright, I'm listening… (recording now)")
            voice.record_audio()
            _thinking_dots("Transcribing your voice")
            query = voice.transcribe_audio()
            if not query:
                _warn("Couldn't catch that. Mind trying again?")
                continue
            print(f'\n  I heard: "{query}"')

        elif choice == "3":
            image_path = input("\n  Image path: ").strip()
            if not os.path.exists(image_path):
                _warn(f'Hmm, I can\'t find a file at "{image_path}". Double-check the path?')
                continue
            user_query = input("  What would you like to know about it? ").strip()
            _thinking_dots("Analysing the image")
            image_description = image_handler.analyze_image(
                chatbot.llm, image_path, user_query
            )
            print(f"\n  Image summary:\n  {image_description}\n")
            query = (
                f"Image Description:\n{image_description}\n\n"
                f"User Question:\n{user_query}"
            )

        elif choice == "4":
            run_ingest(embedding_model, retrieval)
            continue                        # back to menu, no query to process

        elif choice == "5":

            retrieval.close()
            
            _typing(
                "\n  Thanks for using the Research Assistant. "
                "Good luck with your research!\n"
            )
            break

        else:
            _warn("That option doesn't exist — pick 1, 2, 3, 4, or 5.")
            continue

        # ── Retrieve ───────────────────────────────────────────
        _thinking_dots("Searching the knowledge base")
        retrieved_docs = retrieval.retrieve_documents(query=query, top_k=10)
        filtered_docs  = retrieved_docs

        if not filtered_docs:
            _section("NO MATCH FOUND")
            _typing(
                "  I searched through all the uploaded research papers\n"
                "  but couldn't find anything relevant to your question.\n"
                "  Try rephrasing, or check that the right documents are ingested."
            )
            continue

        # ── Build prompt & generate ────────────────────────────
        prompt = retrieval.build_prompt(query, filtered_docs)

        user_question = (
            query.split("User Question:")[-1].strip()
            if "User Question:" in query
            else query
        )

        response = chatbot.generate_response(prompt, original_question=user_question)

        # ── Print answer ───────────────────────────────────────
        _section("ANSWER")
        _typing(response, delay=0.008)

        # ── Optional TTS ───────────────────────────────────────
        print()
        read_aloud = input("  Want me to read that out? [y/n] ").strip().lower()
        if read_aloud == "y":
            _typing("Reading…")
            speaker.speak(response)