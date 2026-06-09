# Multimodal Research RAG

A local web application for searching and asking questions about drone research papers and documents. It lets you type queries, record voice questions, or upload images (like schematics or photos of drones). The system retrieves relevant documentation from a vector database and answers using an ensemble of local models.

The Rag is built using Langchain and Ollama models
The backend is built with FastAPI and the frontend is standard HTML, CSS, and plain JavaScript (no React or build steps needed) using Google Antigravity.

## Key Features

* **LLM Ensemble and Consensus**: Runs your query through three separate models (Mistral, Llama 3, and Qwen) and uses a fourth model (Mistral) as a judge to merge the facts, catch contradictions, and output a clean, single response.
* **Conversational Context**: Remembers previous turns in the chat. If you ask a follow-up question like "how much does it weigh?", the system rephrases it behind the scenes into a standalone query (e.g. "What is the weight of the Hubsan drone?") before searching the database.
* **Image Queries**: You can upload images. The system uses a local vision model (Qwen 2.5-VL or Llava) to generate a text description of the image, then queries the vector index using that description.
* **Voice Transcription**: Record voice queries directly from your browser. The frontend records the audio, converts it to a 16kHz mono WAV file in JavaScript, and sends it to the backend to be transcribed via faster-whisper.
* **Text-to-Speech**: An optional feature on the frontend that reads responses aloud using the browser's native speech synthesis API.
* **Automatic Re-indexing**: The backend watches the document folder. Uploading or deleting files from the web interface triggers a background script to rebuild the Chroma database index automatically.

## Tech Stack

* **Frontend**: HTML5, CSS (with a custom dark glassmorphism design), and vanilla JavaScript.
* **Backend and Logic**: Langchain, PyPDF/PyMuPDF for reading documents, ChromaDB for vector storage, FastAPI.
* **Models**:
  * Embedding Model: MiniLM-L6 (via sentence-transformers).
  * Chat/Ensemble: Ollama (running Mistral, Llama 3, Qwen).
  * Vision: Ollama (running Qwen 2.5-VL or Llava).
  * Speech-to-Text: faster-whisper.

## Prerequisites

You need the following installed on your machine:

1. **Python 3.10+**
2. **Ollama**: Download and run Ollama locally. You will need to pull these models:
   ```bash
   ollama pull mistral
   ollama pull llama3
   ollama pull qwen3:4b
   ollama pull qwen2.5vl:3b
   ```
   *(If you prefer, you can pull `llava` as a fallback vision model instead of qwen2.5vl:3b)*
3. **FFmpeg**: Required on your system path so the whisper library can process audio files.

## Installation and Run Guide

1. **Clone the project repository** and enter the directory:
   ```bash
   git clone <repository_url>
   cd Multimodal-Research-RAG
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Create environment
   python -m venv .venv

   # Activate on Windows
   .venv\Scripts\Activate.ps1

   # Activate on Linux/macOS
   source .venv/bin/activate
   ```

3. **Install the python packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root folder:
   ```env
   DB_PATH=<Database Folder Path>     # Ex. D:\DRDO PROJECT\Multimodal Research RAG\ChromaDB
   DATA_FOLDER=<Data Folder Path>     # Ex. D:\DRDO PROJECT\Multimodal Research RAG\Data
   HOST=127.0.0.1
   PORT=8000
   ```
   Make sure the paths match where you want database files and document uploads to live on your machine.

5. **Start the application**:
   * **To run the Web UI**:
     ```bash
     cd src
     python -m uvicorn app:app --host 127.0.0.1 --port 8000
     ```
     Open `http://127.0.0.1:8000` in your browser. On the first load, wait for the loading screen to complete as it spins up the local models.
   * **To run the Terminal Client**:
     ```bash
     cd src
     python main.py
     ```

6. **Stop the application**:
   * **Web UI or Terminal Client**: Go to the terminal window where the process is active and press `Ctrl + C` to stop the running server or script.

## Project Directory Map

```text
├── ChromaDB/                # Persisted ChromaDB Vector store directory
├── Data/                    # Ingested documents (PDFs, wav, images)
├── requirements.txt         # Required python packages
├── .env                     # Local configuration paths
└── src/
    ├── app.py               # FastAPI backend web app
    ├── main.py              # CLI loop entrypoint
    ├── Retriver.py          # Vector search query and prompt construction
    ├── embedding_pipeline.py# Embedding generation and text splitter
    ├── parser.py            # Reads PDFs, WAVs, and images
    ├── Image_Input.py       # Interacts with local vision models
    ├── Image_Parser.py      # Vision parser for library image indexing
    ├── Voice_input.py       # Helper tools for transcription and mic recording
    └── static/              # Web assets
        ├── index.html       # UI layout
        ├── style.css        # Stylesheet
        └── app.js           # Client logic & voice recorder
```

## Troubleshooting

* **Microphone not working**: Make sure you have allowed microphone permissions for `http://127.0.0.1:8000` in your browser settings.
* **Slow response times**: Since all models run locally, response speed depends on your CPU/GPU hardware. Check if your GPU is running the Ollama processes to verify acceleration.
* **Database lock / Ingestion errors**: If you encounter errors loading the index, delete the `ChromaDB` directory and upload files again via the UI to trigger a clean re-ingest.
