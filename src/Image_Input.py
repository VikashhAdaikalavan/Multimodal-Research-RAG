from PIL import Image
import base64
from langchain_core.messages import HumanMessage
class ImageInput:
    def image_to_base64(
    self,
    image_path
    ):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(
            image_file.read()
            ).decode("utf-8")
    def analyze_image(
        self,
        llm,
        image_path,
        user_query=""
    ):
        image = Image.open(image_path)
        prompt = f"""
        Analyze this image.
        User Query:
        {user_query}
        Describe:
        1. What object is visible.
        2. Drone type if recognizable.
        3. Important visual features.
        4. Describe verbally only the things which you can see in the image after identifying the drone, and don't explain factual details which can't be directly inferred just by seeing.
        """
        image_base64 = self.image_to_base64(
        image_path
        )
        message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": (
                f"data:image/jpeg;base64,{image_base64}"
                )
            }
        ]
        )
        response = llm.invoke([message])
        return response.content