from langchain_community.document_loaders import PyPDFLoader
from Audio_Parser import AudioParser
from Image_Parser import ImageParser
import os

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def load_all_documents(data_folder: str = None):
    """
    Walks the data folder and loads every supported file type:
      • PDF   → text via PyPDFLoader
      • WAV   → transcription via Whisper (AudioParser)
      • Image → visual description via Ollama llava (ImageParser)

    All three return plain LangChain Documents, so the rest of the
    pipeline (chunking, embedding, ChromaDB) works without any changes.
    """

    if data_folder is None:
        data_folder = os.getenv("DATA_FOLDER", r"D:\DRDO PROJECT\RAG Assistant\Data")

    all_documents = []

    # ── collect file paths by type ──────────────────────────────
    pdf_files   = []
    wav_files   = []
    image_files = []

    for filename in os.listdir(data_folder):
        ext = os.path.splitext(filename)[1].lower()
        full_path = os.path.join(data_folder, filename)
        if ext == ".pdf":
            pdf_files.append(full_path)
        elif ext == ".wav":
            wav_files.append(full_path)
        elif ext in IMAGE_EXTENSIONS:
            image_files.append(full_path)

    # ── PDFs ────────────────────────────────────────────────────
    if pdf_files:
        print(f"\nLoading {len(pdf_files)} PDF file(s)...")
        for pdf_path in pdf_files:
            loader = PyPDFLoader(pdf_path)
            docs   = loader.load()
            all_documents.extend(docs)
            print(f"  [OK] {os.path.basename(pdf_path)}  ({len(docs)} page(s))")

    # ── WAV audio ───────────────────────────────────────────────
    if wav_files:
        print(f"\nTranscribing {len(wav_files)} audio file(s)...")
        audio_parser = AudioParser()
        audio_parser.load_model()
        for wav_path in wav_files:
            docs = audio_parser.parse_audio(wav_path)
            all_documents.extend(docs)
            print(f"  [OK] {os.path.basename(wav_path)}")

    # ── Images ──────────────────────────────────────────────────
    if image_files:
        print(f"\nDescribing {len(image_files)} image file(s) via vision model...")
        image_parser = ImageParser()
        image_parser.load_model()
        for img_path in image_files:
            try:
                docs = image_parser.parse_image(img_path)
                all_documents.extend(docs)
                print(f"  {os.path.basename(img_path)}")
            except Exception as exc:
                print(f" Skipped {os.path.basename(img_path)}: {exc}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents
