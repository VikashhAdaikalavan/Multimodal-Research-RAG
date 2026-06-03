from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def create_chunks(documents):
    splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    ) 
    chunked_documents = splitter.split_documents(documents)
    return chunked_documents

class EmbeddingModel:
    def __init__(self):
        self.embedding_model = None
    def load_model(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    def get_embedding_model(self):
        return self.embedding_model

def create_embeddings(chunks, embedding_model):
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.page_content)
    embeddings = embedding_model.embed_documents(chunk_texts)
    return embeddings
