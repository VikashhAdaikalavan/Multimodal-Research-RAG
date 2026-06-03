import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

class VoiceInput:
    def __init__(self):
        self.model = None

    def load_model(self): 
        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
        print("Whisper model loaded.")

    def record_audio(
        self,
        duration=5, 
        sample_rate=16000 
    ): 
        print("\nSpeak now...\n")
        
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1
        )
        sd.wait() 
        write(
            "query.wav",
            sample_rate,
            audio
        )
        print("Recording complete.")

    def transcribe_audio(self):
        segments, info = self.model.transcribe(
            "query.wav"
        )

        query = ""
        for segment in segments:
            query += segment.text
        return query.strip()
