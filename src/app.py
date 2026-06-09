import os
import sys

# Add the src folder to system path to resolve local imports cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import threading
import gc
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json

# Load environment configuration
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Set up paths
DB_PATH = os.getenv("DB_PATH", r"D:\DRDO PROJECT\RAG Assistant\ChromaDB")
DATA_FOLDER = os.getenv("DATA_FOLDER", r"D:\DRDO PROJECT\RAG Assistant\Data")

# Ensure folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

app = FastAPI(title="Multimodal Research RAG API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
embedding_model = None
retrieval = None
chatbot = None
voice = None
image_handler = None
vision_llm = None

# System loading state
state = {
    "status": "loading",
    "steps": {
        "embeddings": "pending",
        "database": "pending",
        "llm_ensemble": "pending",
        "whisper": "pending"
    },
    "error_message": ""
}

# Ingestion state tracking
ingest_state = {
    "status": "idle",  # "idle", "running", "success", "error"
    "phase": "",       # Progress description
    "message": "",     # Detailed feedback
    "timestamp": 0
}

# Debouncer variables for auto-ingestion
ingest_timer = None
ingest_timer_lock = threading.Lock()

def trigger_auto_ingest_debounced():
    global ingest_timer
    with ingest_timer_lock:
        if ingest_timer is not None:
            ingest_timer.cancel()
            
        # Start a new timer to execute run_auto_ingest_task after 2 seconds of inactivity
        ingest_timer = threading.Timer(2.0, run_auto_ingest_task)
        ingest_timer.start()

def run_auto_ingest_task():
    global ingest_state
    # Check if backend models are loaded, and if database is already ingesting
    if state["status"] != "ready":
        print("Auto-ingest deferred: ML models are still loading.")
        return
        
    if ingest_state["status"] == "running":
        print("Auto-ingest deferred: Ingestion is already running. Retrying in 5 seconds...")
        global ingest_timer
        with ingest_timer_lock:
            ingest_timer = threading.Timer(5.0, run_auto_ingest_task)
            ingest_timer.start()
        return

    print("Auto-ingest triggered...")
    thread = threading.Thread(target=run_ingest_bg)
    thread.daemon = True
    thread.start()


def load_resources_bg():
    global embedding_model, retrieval, chatbot, voice, image_handler, vision_llm, state
    try:
        print("Starting background initialization of ML models...")
        
        # 1. Load Embeddings
        state["steps"]["embeddings"] = "loading"
        from embedding_pipeline import EmbeddingModel
        embedding = EmbeddingModel()
        embedding.load_model()
        embedding_model = embedding.get_embedding_model()
        state["steps"]["embeddings"] = "ready"
        print("[OK] Embeddings model loaded.")
        
        # 2. Load Retriever & DB
        state["steps"]["database"] = "loading"
        from Retriver import Retriever
        retrieval = Retriever()
        
        # Check if DB files exist
        db_exists = os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0
        if not db_exists:
            state["steps"]["database"] = "empty"
            print("[WARN] Vector database directory is empty. Ingestion required.")
        else:
            retrieval.load_vector_store(embedding_model)
            state["steps"]["database"] = "ready"
            print("[OK] Vector database loaded successfully.")
            
        # 3. Load LLM Ensemble
        state["steps"]["llm_ensemble"] = "loading"
        from main import Chatbot
        chatbot = Chatbot()
        chatbot.load_llm()
        state["steps"]["llm_ensemble"] = "ready"
        print("[OK] Ollama LLM ensemble ready.")
        
        # 4. Load Whisper Voice model
        state["steps"]["whisper"] = "loading"
        from Voice_input import VoiceInput
        voice = VoiceInput()
        voice.load_model()
        state["steps"]["whisper"] = "ready"
        print("[OK] Whisper audio transcribers ready.")
        
        from Image_Input import ImageInput
        image_handler = ImageInput()
        
        from langchain_ollama import ChatOllama
        # Load vision model (qwen2.5vl:3b or fallback to llava)
        try:
            print("Spinning up vision model: qwen2.5vl:3b...")
            vision_llm = ChatOllama(model="qwen2.5vl:3b", temperature=0.1)
            # Test invocation to check if model works / exists
            vision_llm.invoke("Hi")
            print("[OK] Vision model qwen2.5vl:3b loaded.")
        except Exception as e:
            print(f"[WARN] Failed to load qwen2.5vl:3b ({e}). Trying llava:latest...")
            try:
                vision_llm = ChatOllama(model="llava:latest", temperature=0.1)
                vision_llm.invoke("Hi")
                print("[OK] Vision model llava:latest loaded.")
            except Exception as e2:
                print(f"[ERROR] Failed to load llava:latest ({e2}). Vision features might not function.")
                vision_llm = None
        
        state["status"] = "ready"
        print("[OK] All models loaded successfully. Web app backend is fully operational!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        state["status"] = "error"
        state["error_message"] = str(e)
        print(f"Error loading resources: {e}")


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_resources_bg)
    thread.daemon = True
    thread.start()


@app.get("/api/status")
def get_status():
    db_exists = os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0
    return {
        "status": state["status"],
        "steps": state["steps"],
        "error_message": state["error_message"],
        "db_path": DB_PATH,
        "db_exists": db_exists,
        "data_folder": DATA_FOLDER
    }


@app.get("/api/documents")
def get_documents():
    if not os.path.exists(DATA_FOLDER):
        return []
    files = []
    for filename in os.listdir(DATA_FOLDER):
        filepath = os.path.join(DATA_FOLDER, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            ext = os.path.splitext(filename)[1].lower()
            filetype = "unknown"
            if ext == ".pdf":
                filetype = "pdf"
            elif ext == ".wav":
                filetype = "audio"
            elif ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
                filetype = "image"
            
            files.append({
                "name": filename,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "type": filetype
            })
    return files


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename)
    dest_path = os.path.join(DATA_FOLDER, filename)
    
    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    trigger_auto_ingest_debounced()
    return {"message": f"Successfully uploaded {filename}", "filename": filename}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    filename = os.path.basename(filename)
    filepath = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            trigger_auto_ingest_debounced()
            return {"message": f"Successfully deleted {filename}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="File not found")


def condense_query(llm, history_json: str, query: str) -> str:
    if not history_json:
        return query
    try:
        history = json.loads(history_json)
        if not history or not isinstance(history, list):
            return query
            
        # Format history turns for context
        history_str = ""
        # Only take the last 10 messages (5 turns) to keep context concise
        for msg in history[-10:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if "Voice Recording sent" in content:
                continue
            history_str += f"{role}: {content}\n"
            
        if not history_str.strip():
            return query

        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rephrase the follow-up question to be a standalone question (in English) "
            "that can be used to query a vector database. Do NOT answer the question, "
            "just output the standalone question and nothing else.\n\n"
            f"Conversation History:\n{history_str}\n"
            f"Follow-up Question: {query}\n"
            "Standalone Question:"
        )
        response = llm.invoke(prompt)
        condensed = response.content.strip()
        if (condensed.startswith('"') and condensed.endswith('"')) or (condensed.startswith("'") and condensed.endswith("'")):
            condensed = condensed[1:-1].strip()
        print(f"Conversational Context - Rephrased: '{query}' -> '{condensed}'")
        return condensed
    except Exception as e:
        print(f"Failed to condense query with history: {e}")
        return query


@app.post("/api/query")
async def query_endpoint(
    query: str = Form(...),
    image: UploadFile = File(None),
    history: str = Form(None)
):
    global embedding_model, retrieval, chatbot, image_handler, vision_llm
    
    if state["status"] != "ready":
        raise HTTPException(status_code=503, detail="System is still loading models. Please wait.")
        
    if not retrieval or retrieval.vector_store is None:
        raise HTTPException(status_code=400, detail="Vector database not loaded. Please ingest documents first.")
        
    temp_image_path = None
    query_processed = query
    image_analysis = None
    
    try:
        # Condense query using conversational history
        if history:
            query_processed = condense_query(chatbot.llm, history, query)
            
        # If image is uploaded
        if image:
            temp_dir = os.path.join(os.path.dirname(DATA_FOLDER), "temp_query_images")
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, f"query_{int(time.time())}_{image.filename}")
            
            with open(temp_image_path, "wb") as f:
                content = await image.read()
                f.write(content)
                
            # Perform image analysis using the vision LLM
            if vision_llm is None:
                raise Exception("Vision model (qwen2.5vl:3b / llava:latest) is not loaded or available on the backend.")
            image_description = image_handler.analyze_image(
                vision_llm, temp_image_path, query
            )
            image_analysis = image_description
            query_processed = (
                f"Image Description:\n{image_description}\n\n"
                f"User Question:\n{query}"
            )
            
        # Retrieve documents from vector database
        retrieved_docs = retrieval.retrieve_documents(query=query_processed, top_k=10)
        
        if not retrieved_docs:
            return {
                "answer": "I searched through all the uploaded research papers but couldn't find anything relevant to your question.",
                "sources": [],
                "image_description": image_analysis
            }
            
        # Build prompt & generate response
        prompt = retrieval.build_prompt(query_processed, retrieved_docs)
        
        user_question = (
            query_processed.split("User Question:")[-1].strip()
            if "User Question:" in query_processed
            else query_processed
        )
        
        response = chatbot.generate_response(prompt, original_question=user_question)
        
        # Format sources
        sources = []
        for doc, score in retrieved_docs:
            sources.append({
                "content": doc.page_content,
                "score": float(score),
                "source": doc.metadata.get("source", "Unknown"),
                "filename": os.path.basename(doc.metadata.get("source", "")) or doc.metadata.get("file", "Unknown"),
                "type": doc.metadata.get("type", "text")
            })
            
        return {
            "answer": response,
            "sources": sources,
            "image_description": image_analysis
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temp image file if created
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass


@app.post("/api/voice-query")
async def voice_query_endpoint(
    audio: UploadFile = File(...),
    history: str = Form(None)
):
    global voice, embedding_model, retrieval, chatbot
    
    if state["status"] != "ready":
        raise HTTPException(status_code=503, detail="System is still loading models. Please wait.")
        
    if not retrieval or retrieval.vector_store is None:
        raise HTTPException(status_code=400, detail="Vector database not loaded. Please ingest documents first.")
        
    temp_dir = os.path.join(os.path.dirname(DATA_FOLDER), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    temp_wav_path = os.path.join(temp_dir, f"query_{int(time.time())}.wav")
    
    try:
        with open(temp_wav_path, "wb") as f:
            content = await audio.read()
            f.write(content)
            
        # Transcribe audio using Whisper
        segments, info = voice.model.transcribe(temp_wav_path)
        query = "".join(segment.text for segment in segments).strip()
        
        if not query:
            return {
                "error": "Could not transcribe audio.",
                "transcription": "",
                "answer": "Sorry, I couldn't understand the audio or the recording was empty. Please try again."
            }
            
        # Condense transcribed query using conversational history
        query_processed = query
        if history:
            query_processed = condense_query(chatbot.llm, history, query)
            
        # Now run RAG on processed query text
        retrieved_docs = retrieval.retrieve_documents(query=query_processed, top_k=10)
        
        if not retrieved_docs:
            return {
                "transcription": query,
                "answer": "I searched through all the uploaded research papers but couldn't find anything relevant to your question.",
                "sources": []
            }
            
        prompt = retrieval.build_prompt(query_processed, retrieved_docs)
        response = chatbot.generate_response(prompt, original_question=query)
        
        sources = []
        for doc, score in retrieved_docs:
            sources.append({
                "content": doc.page_content,
                "score": float(score),
                "source": doc.metadata.get("source", "Unknown"),
                "filename": os.path.basename(doc.metadata.get("source", "")) or doc.metadata.get("file", "Unknown"),
                "type": doc.metadata.get("type", "text")
            })
            
        return {
            "transcription": query,
            "answer": response,
            "sources": sources
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass


def run_ingest_bg():
    global embedding_model, retrieval, ingest_state
    ingest_state["status"] = "running"
    ingest_state["timestamp"] = time.time()
    
    try:
        from parser import load_all_documents
        from embedding_pipeline import create_chunks
        from vectordb import VectorDatabase
        
        # 1 — Parse all PDFs / WAVs / Images
        ingest_state["phase"] = "Parsing documents (PDFs, WAV transcriptions, Visual image details)..."
        docs = load_all_documents(DATA_FOLDER)
        if not docs:
            ingest_state["status"] = "success"
            ingest_state["phase"] = "Done"
            ingest_state["message"] = "No documents found in the data folder. Nothing to ingest."
            return
            
        # 2 — Chunk documents
        ingest_state["phase"] = f"Splitting {len(docs)} files into overlapping text chunks..."
        chunks = create_chunks(docs)
        
        # 3 — Rebuild ChromaDB
        ingest_state["phase"] = f"Writing {len(chunks)} text chunks to ChromaDB (Generating embeddings)..."
        
        # Safely shut down database locks
        retrieval.release()
        retrieval.close()
        gc.collect()
        time.sleep(2)
        
        vectordb = VectorDatabase(DB_PATH)
        vectordb.create_vector_store(chunks, embedding_model)
        vectordb.close()
        del vectordb
        gc.collect()
        
        # 4 — Hot-swap the retriever live
        ingest_state["phase"] = "Hot-swapping vector database and reloading index..."
        retrieval.load_vector_store(embedding_model)
        
        ingest_state["status"] = "success"
        ingest_state["phase"] = "Completed"
        ingest_state["message"] = f"Knowledge base rebuilt successfully with {len(chunks)} chunks!"
        print("[OK] Vector DB hot-swapped successfully.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        ingest_state["status"] = "error"
        ingest_state["message"] = str(e)
        print(f"Ingestion failed: {e}")
        # Try to reload the old database if possible
        try:
            retrieval.load_vector_store(embedding_model)
        except:
            pass


@app.post("/api/ingest")
def trigger_ingest(background_tasks: BackgroundTasks):
    global ingest_state
    if ingest_state["status"] == "running":
        return {"message": "Ingestion is already running.", "status": "running"}
        
    ingest_state["status"] = "running"
    ingest_state["phase"] = "Initiating parser pipeline"
    ingest_state["message"] = ""
    ingest_state["timestamp"] = time.time()
    
    background_tasks.add_task(run_ingest_bg)
    return {"message": "Ingestion started.", "status": "running"}


@app.get("/api/ingest/status")
def get_ingest_status():
    return ingest_state


# Serve static frontend files
from fastapi.staticfiles import StaticFiles

static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
