from faster_whisper import WhisperModel
from langchain_core.documents import Document

class AudioParser:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

    def parse_audio(self, file_path):
        segments, info = self.model.transcribe(
            file_path
        )
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
        document = Document(
            page_content=transcript,
            metadata={
                "source": file_path,
                "type": "audio"
            }
        )
        return [document]