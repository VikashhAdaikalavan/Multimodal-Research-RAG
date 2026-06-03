import base64
import os
import io
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from PIL import Image


class ImageParser:
    """
    Converts image files into LangChain Documents by asking a vision-capable
    Ollama model to describe them in detail.  The description becomes the
    page_content that gets chunked and embedded just like PDF or audio text.
    """

    # llava is the standard vision model available via Ollama
    VISION_MODEL = "llava"

    def __init__(self):
        self.llm = None

    def load_model(self):
        self.llm = ChatOllama(model=self.VISION_MODEL,temperature=0,num_ctx=2048)
        print(f"  Vision model ready  →  {self.VISION_MODEL}")

    
    def _image_to_base64(self, image_path: str) -> str:
        img = Image.open(image_path)
        print(f"Image size: {img.width}x{img.height}")

        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((1024, 1024))

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _describe_image(self, image_path: str) -> str:
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".gif":  "image/gif",
            ".webp": "image/webp",
            ".bmp":  "image/bmp",
        }
        mime = mime_map.get(ext, "image/jpeg")
        image_b64 = self._image_to_base64(image_path)

        prompt = """You are a technical document analyser specialising in drone and aerospace systems.

        Describe this image in full detail for the purpose of building a searchable knowledge base.
        Cover ALL of the following that apply:

        1. Overall scene / subject — what is shown?
        2. Colour, model markings, or identifying features (if any).
        3. Components visible: rotors, arms, frame, cameras, sensors, payload, landing gear, etc.
        4. Text, labels, callouts, or annotations present in the image.
        5. Diagrams, charts, schematics, or graphs — describe axes, values, legends, and what they communicate.
        6. Technical specifications or numerical data visible.
        7. Any context clues: environment, scale, colour coding, connectors, materials.

        Be factual and exhaustive. Use technical vocabulary. Do NOT speculate beyond what is visible.
        """

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": f"data:{mime};base64,{image_b64}"}
        ])

        try:
            response = self.llm.invoke([message])
            return response.content.strip()

        except Exception as e:
            print(f"Vision processing failed: {e}")
            return ""

    def parse_image(self, image_path: str) -> list[Document]:
        """
        Returns a single-element list containing a Document whose
        page_content is the vision-model description of the image.
        """
        description = self._describe_image(image_path)
        doc = Document(
            page_content=description,
            metadata={
                "source": image_path,
                "type":   "image",
                "file":   os.path.basename(image_path),
            }
        )
        return [doc]