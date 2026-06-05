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
    VISION_MODEL = "qwen2.5vl:3b"

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
        img.thumbnail((512, 512))

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

        prompt = """Analyze this image and identify any machine, equipment, vehicle, or technical system present.

        Provide:

        Machine name and type (We ned to know the name of teh machine / equipment).
        Manufacturer, model, or designation if visible.
        Major visible components and their functions.
        Machine purpose and likely application.
        All visible text, labels, logos, and markings.
        Physical appearance, color, shape, and distinguishing features.
        Environment and operating context.
        Condition (new, used, damaged, etc.).
        20-30 technical keywords for search and retrieval.

        Focus on factual observations. If identification is uncertain, provide the most likely possibilities and state your confidence.
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
        print(doc)
        return [doc]